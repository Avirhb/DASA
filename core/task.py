"""ABD models, separate data streams, and native-resolution evaluation."""

import csv
import importlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from .io import read_json, resolve_asset


def load_backend(root):
    code = Path(root).resolve() / "code"
    if not (code / "utils/displacement.py").is_file():
        raise FileNotFoundError(f"ABD code missing: {code}")
    for name in ("networks", "utils", "dataloaders"):
        if name in sys.modules:
            module = sys.modules[name]
            locations = list(getattr(module, "__path__", []))
            if getattr(module, "__file__", None):
                locations.append(module.__file__)
            if not locations or not all(
                (Path(p).resolve().is_relative_to(code) for p in locations)
            ):
                raise RuntimeError(
                    f"ABD import collision for {name}; use a fresh CLI process"
                )
    sys.path.insert(0, str(code))
    return SimpleNamespace(
        WeakStrongAugment=importlib.import_module(
            "dataloaders.dataset"
        ).WeakStrongAugment,
        UNet=importlib.import_module("networks.unet").UNet,
        SwinUnet=importlib.import_module("networks.vision_transformer").SwinUnet,
        default_config=importlib.import_module("networks.config")._C,
        losses=importlib.import_module("utils.losses"),
        ramps=importlib.import_module("utils.ramps"),
        ABD_I=importlib.import_module("utils.displacement").ABD_I,
        ABD_R=importlib.import_module("utils.displacement").ABD_R,
    )


def make_models(config, paths, backend, device):
    import torch
    from yacs.config import CfgNode

    training = config["training"]
    architecture = backend.default_config.clone()
    architecture.defrost()
    architecture.DATA.BATCH_SIZE = training["batch_size"]
    architecture.DATA.CACHE_MODE = "part"
    architecture.AMP_OPT_LEVEL = "O1"
    architecture.merge_from_other_cfg(CfgNode(read_json(config["swin_config"])))
    architecture.MODEL.PRETRAIN_CKPT = paths["swin_checkpoint"]
    architecture.freeze()
    if not Path(paths["swin_checkpoint"]).is_file():
        raise FileNotFoundError(paths["swin_checkpoint"])
    unet = backend.UNet(in_chns=1, class_num=training["num_classes"]).to(device)
    swin = backend.SwinUnet(
        architecture,
        img_size=[training["image_size"]] * 2,
        num_classes=training["num_classes"],
    ).to(device)
    probe = "swin_unet.patch_embed.proj.weight"
    before = swin.state_dict()[probe].detach().cpu().clone()
    swin.load_from(architecture)
    if torch.equal(before, swin.state_dict()[probe].detach().cpu()):
        raise RuntimeError("Swin pretrained weights did not load")
    return (unet, swin)


class TrainDataset(Dataset):

    def __init__(self, records, labeled, roots, config, augment):
        self.records = records
        self.labeled = labeled
        self.roots = roots
        self.threshold = config["mask_threshold"]
        self.augment = augment([config["image_size"]] * 2)
        if not labeled and any((r["mask_path"] for r in records)):
            raise ValueError("Unlabeled data may not contain mask paths")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        row = self.records[i]
        with Image.open(resolve_asset(row["image_path"], self.roots)) as im:
            image = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        if self.labeled:
            with Image.open(resolve_asset(row["mask_path"], self.roots)) as im:
                label = (np.asarray(im.convert("L")) > self.threshold).astype(np.uint8)
            if label.shape != image.shape:
                raise ValueError("Labeled image/mask shape mismatch")
        else:
            label = np.zeros(image.shape, dtype=np.uint8)
        return self.augment({"image": image, "label": label})


class TestDataset(Dataset):

    def __init__(self, records, roots, config):
        self.records = records
        self.roots = roots
        self.config = config

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        row = self.records[i]
        with Image.open(resolve_asset(row["image_path"], self.roots)) as im:
            original_size = im.size
            image = im.convert("L").resize(
                (self.config["image_size"],) * 2, Image.Resampling.BILINEAR
            )
            tensor = torch.from_numpy(
                np.asarray(image, dtype=np.float32) / 255.0
            ).unsqueeze(0)
        with Image.open(resolve_asset(row["mask_path"], self.roots)) as im:
            if im.size != original_size:
                raise ValueError("Test image/mask shape mismatch")
            target = torch.from_numpy(
                (np.asarray(im.convert("L")) > self.config["mask_threshold"]).astype(
                    np.uint8
                )
            )
        return (tensor, target, row["image_id"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, workers, seed, drop_last=False):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def forever(loader):
    while True:
        yield from loader


def state_dict_to_cpu(model):
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def evaluate_test(model, loader, model_name, output_dir):
    prediction_dir = output_dir / f"test_predictions_{model_name}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    model.eval()
    with torch.no_grad():
        for image, target, stem in loader:
            target_np = target.squeeze(0).numpy().astype(bool)
            logits = model(image.to(next(model.parameters()).device))
            logits = F.interpolate(
                logits, size=target_np.shape, mode="bilinear", align_corners=False
            )
            prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(bool)
            intersection = int(np.logical_and(prediction, target_np).sum())
            pred_count = int(prediction.sum())
            target_count = int(target_np.sum())
            denominator = pred_count + target_count
            union = pred_count + target_count - intersection
            dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
            iou = 1.0 if union == 0 else intersection / union
            Image.fromarray(prediction.astype(np.uint8) * 255).save(
                prediction_dir / f"{stem[0]}.png"
            )
            rows.append(
                {
                    "image_id": stem[0],
                    "dice": dice,
                    "iou": iou,
                    "prediction_pixels": pred_count,
                    "target_pixels": target_count,
                }
            )
    with open(
        output_dir / f"per_case_metrics_{model_name}.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dice_values = np.asarray([row["dice"] for row in rows], dtype=float)
    iou_values = np.asarray([row["iou"] for row in rows], dtype=float)
    return {
        "test_count": len(rows),
        "mean_dice": float(dice_values.mean()),
        "std_dice": float(dice_values.std(ddof=1)) if len(rows) > 1 else 0.0,
        "median_dice": float(np.median(dice_values)),
        "mean_iou": float(iou_values.mean()),
        "cases_dice_gt_085": int((dice_values > 0.85).sum()),
        "cases_dice_gt_090": int((dice_values > 0.9).sum()),
        "cases_dice_gt_095": int((dice_values > 0.95).sum()),
    }
