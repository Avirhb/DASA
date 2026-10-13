"""One-shot DASA/IGFS inference. Run once into a new output directory."""

import argparse
import csv
import random
from pathlib import Path
from core.io import load_config, write_json


def run(config, paths, dataset):
    import numpy as np
    import torch
    from PIL import Image
    from modelscope import AutoModel
    from third_party.segment_anything import sam_model_registry, SamPredictor
    from core.dasa import DASA
    from core.igfs import IGFS
    from core.numerics import (
        load_query,
        set_reproducibility,
        compute_iou_dice,
        compute_auc_metrics,
    )

    dataset = dataset.upper()
    entry = config["datasets"][dataset]
    algorithm = dict(config["algorithm"], **entry["parameters"])
    algorithm.update(entry.get("post_processing", {}))
    seed = config["seed_overrides"].get(dataset, config["seed"])
    dirs = paths.get("dataset_directories", {}).get(dataset)
    images = (
        Path(dirs["images"]) if dirs else Path(paths["data_root"]) / dataset / "images"
    )
    masks = (
        Path(dirs["masks"]) if dirs else Path(paths["data_root"]) / dataset / "masks"
    )
    files = sorted(
        p.name
        for p in images.iterdir()
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if len(files) < 2 or any(not (masks / n).is_file() for n in files):
        raise ValueError("Need at least two images and matching same-name masks")
    support = random.Random(seed).sample(files, 1)[0]
    queries = [n for n in files if n != support]
    limit = config["execution"]["max_images"]
    if limit is not None and limit < 1:
        raise ValueError("max_images must be null or positive")
    selected = queries[:limit]
    output = Path(config["output_dir"]) / dataset / str(seed)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    model_config = config["models"]
    device = model_config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = model_config["dtype"]
    if dtype == "auto":
        dtype = (
            "bfloat16"
            if device.startswith("cuda") and torch.cuda.is_bf16_supported()
            else "float32"
        )
    variant = config["variant"]
    if variant not in ("pbas", "igfs") or (variant == "igfs" and dtype != "float32"):
        raise ValueError("Expected PBAS or FP32-query IGFS")
    set_reproducibility(algorithm["kmeans_seed"])
    torch.set_num_threads(config["execution"]["torch_threads"])
    engine = (DASA if variant == "pbas" else IGFS)(
        algorithm, device, getattr(torch, dtype)
    )
    dino = (
        AutoModel.from_pretrained(
            paths["dinov3_model"],
            trust_remote_code=model_config["trust_remote_code"],
            local_files_only=model_config["local_files_only"],
        )
        .to(device, dtype=engine.dtype)
        .eval()
    )
    sam = SamPredictor(
        sam_model_registry[model_config["sam_type"]](
            checkpoint=paths["sam_checkpoint"]
        ).to(device)
    )
    if variant == "pbas":
        bank = engine.build_support_banks(
            images, masks, {seed: {"order": (support,)}}, dino
        )[seed]
        sequence = selected
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        bank = engine.build_support_bank(images, masks, support, dino)
        sampling_rng = np.random.RandomState(seed)
        # Include support in the sampling stream, as in the original IGFS runner.
        sequence = files[: files.index(selected[-1]) + 1]
    write_json(
        output / "run_config.json",
        dict(
            config=config,
            dataset=dataset,
            seed=seed,
            support=support,
            total_queries=len(queries),
        ),
    )
    rows = []
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "iou", "dice", "auroc", "auprc"]
        )
        writer.writeheader()
        for name in sequence:
            image, array, gray, gt, width, height = load_query(
                images / name, masks / name
            )
            if variant == "pbas":
                global_features, dense = engine.extract_dino_features(image, dino)
                prob = engine.compute_probability_map(
                    global_features, dense, gray, width, height, bank, 1, name
                )
                sam.set_image(array)
                prediction = engine.predict_mask(prob, sam, width, height)
            else:
                _, dense = engine.extract_reference_dino_features(image, dino)
                prob, _ = engine.compute_igfs_probability(
                    dense,
                    gray,
                    width,
                    height,
                    bank,
                    algorithm["sample_ratio"],
                    algorithm["k_sigma"],
                    dataset,
                    name,
                    algorithm["pos_map_center"],
                    sampling_rng,
                )
                if name == support:
                    continue
                sam.set_image(array)
                prompt = dict(
                    core_ratio=algorithm["bbox_core_ratio"],
                    safe_ratio=algorithm["bbox_safe_ratio"],
                    padding=algorithm["bbox_padding"],
                    num_points=algorithm["num_point_prompts"],
                )
                prediction, _, _ = engine.predict_igfs_mask(
                    prob, sam, width, height, prompt
                )
            iou, dice = compute_iou_dice(prediction, gt)
            auroc, auprc = compute_auc_metrics(prob, gt)
            row = dict(
                image_id=name,
                iou=float(iou),
                dice=float(dice),
                auroc=float(auroc),
                auprc=float(auprc),
            )
            rows.append(row)
            writer.writerow(row)
            f.flush()
            if config["execution"]["save_predictions"]:
                (output / "predictions").mkdir(exist_ok=True)
                Image.fromarray(prediction.astype(np.uint8) * 255).save(
                    output / "predictions" / name, format="PNG"
                )
            print(
                f"{dataset} {len(rows)}/{len(selected)} {name} Dice={dice:.4f}",
                flush=True,
            )
    summary = dict(
        support=support,
        count=len(rows),
        total_queries=len(queries),
        complete=len(rows) == len(queries),
        units="0-1",
        **{
            k: float(np.mean([r[k] for r in rows]))
            for k in ("iou", "dice", "auroc", "auprc")
        },
    )
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(Path(__file__).parent / "configs/dasa.json")
    )
    parser.add_argument("--paths")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    run(*load_config(args.config, args.paths), args.dataset)
