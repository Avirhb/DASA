"""One-shot initiated task learning from existing labeled/unlabeled/test CSVs."""

import argparse
from pathlib import Path
from core.io import load_config, load_manifests, write_csv, write_json


def train(config, paths, smoke=False):
    import torch
    from core.loss import loss_step
    from core.task import load_backend, make_models, TrainDataset, TestDataset
    from core.task import (
        seed_everything,
        make_loader,
        forever,
        state_dict_to_cpu,
        atomic_torch_save,
        evaluate_test,
    )

    t = config["training"]
    device = config["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. For a CPU smoke run, set device=cpu in the config."
        )
    if t["image_size"] != 224 or t["num_classes"] != 2 or t["mask_threshold"] != 128:
        raise ValueError(
            "This model expects binary 224px inputs and mask threshold >128"
        )
    if t["batch_size"] != 2 * t["labeled_bs"] or t["labeled_bs"] < 1:
        raise ValueError("Use equal positive labeled/unlabeled batch sizes")
    if t["iterations"] < 1 or t["log_interval"] < 1 or t["ramp_interval"] < 1:
        raise ValueError("Training intervals must be positive")
    if t["abd_patch_size"] * t["abd_grid_size"] != t["image_size"]:
        raise ValueError("ABD patches must cover the image")
    roles = load_manifests(config["manifest_dir"])
    if any(len(roles[k]) < t["labeled_bs"] for k in ("labeled", "unlabeled")):
        raise ValueError("Not enough labeled/unlabeled cases for the batch size")
    steps = config["smoke_steps"] if smoke else t["iterations"]
    if not 1 <= steps <= t["iterations"]:
        raise ValueError("Invalid step count")
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    torch.set_num_threads(config["torch_threads"])
    seed_everything(config["seed"])
    backend = load_backend(paths["abd_root"])
    unet, swin = make_models(config, paths, backend, device)
    labeled = TrainDataset(
        roles["labeled"],
        True,
        paths.get("asset_roots", {}),
        t,
        backend.WeakStrongAugment,
    )
    unlabeled = TrainDataset(
        roles["unlabeled"],
        False,
        paths.get("asset_roots", {}),
        t,
        backend.WeakStrongAugment,
    )
    batches_l = forever(
        make_loader(
            labeled,
            t["labeled_bs"],
            True,
            config["num_workers"],
            config["seed"] + 1000,
            drop_last=True,
        )
    )
    batches_u = forever(
        make_loader(
            unlabeled,
            t["labeled_bs"],
            True,
            config["num_workers"],
            config["seed"] + 2000,
            drop_last=True,
        )
    )
    optimizers = [
        torch.optim.SGD(
            m.parameters(),
            lr=t["base_lr"],
            momentum=t["momentum"],
            weight_decay=t["weight_decay"],
        )
        for m in (unet, swin)
    ]
    write_json(
        output / "run_config.json",
        dict(
            config=config,
            smoke=smoke,
            selected_branch="swin",
            counts={k: len(v) for k, v in roles.items()},
        ),
    )
    unet.train()
    swin.train()
    history = []
    for iteration in range(1, steps + 1):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss, weight = loss_step(
            unet, swin, next(batches_l), next(batches_u), iteration, device, t, backend
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at iteration {iteration}")
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            for group in optimizer.param_groups:
                group["lr"] = (
                    t["base_lr"] * (1 - iteration / t["iterations"]) ** t["lr_power"]
                )
        if iteration % t["log_interval"] == 0 or iteration == steps:
            row = dict(
                iteration=iteration,
                loss=float(loss.detach()),
                consistency_weight=weight,
            )
            history.append(row)
            write_csv(output / "training.csv", history)
            print(row, flush=True)
    if smoke:
        result = dict(smoke=True, steps=steps, test_evaluations=0)
    else:
        for name, model in (("unet", unet), ("swin", swin)):
            atomic_torch_save(state_dict_to_cpu(model), output / f"last_{name}.pth")
        # The final Swin checkpoint is fixed before any test evaluation.
        test = TestDataset(roles["test"], paths.get("asset_roots", {}), t)
        loader = make_loader(
            test, 1, False, config["num_workers"], config["seed"] + 4000
        )
        result = dict(
            selected_branch="swin",
            steps=steps,
            test=evaluate_test(swin, loader, "swin", output),
        )
    write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(Path(__file__).parent / "configs/task_learning.json")
    )
    parser.add_argument("--paths")
    parser.add_argument(
        "--manifest-dir", help="Override the folder containing the three CSV manifests"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run smoke_steps only, without testing"
    )
    args = parser.parse_args()
    config, paths = load_config(args.config, args.paths)
    if args.manifest_dir:
        config["manifest_dir"] = str(Path(args.manifest_dir).resolve())
    train(config, paths, args.smoke)
