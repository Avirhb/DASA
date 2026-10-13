"""Image-keyed deterministic prototype clustering from the canonical engine."""

import hashlib
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def stable_seed(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def deterministic_torch_kmeans(features, k, seed, n_iter=10):
    count = features.shape[0]
    if count == 0:
        raise ValueError("K-means received an empty feature tensor")
    k = min(k, count)
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    centers = features[
        torch.randperm(count, device=features.device, generator=generator)[:k]
    ].clone()
    for _ in range(n_iter):
        labels = torch.mm(features, centers.t()).argmax(dim=1)
        new_centers = torch.zeros_like(centers)
        for cluster_index in range(k):
            members = features[labels == cluster_index]
            if len(members):
                new_centers[cluster_index] = members.mean(0)
        centers = F.normalize(new_centers, dim=-1)
    return centers


def load_query(image_path, mask_path):
    with Image.open(image_path) as image_handle:
        image = image_handle.convert("RGB")
        width, height = image.size
        image_array = np.array(image)
        image_gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    with Image.open(mask_path) as mask_handle:
        ground_truth = np.array(mask_handle.convert("L")) > 128
    if ground_truth.shape != (height, width):
        raise RuntimeError(
            f"Image/mask shape mismatch for {image_path.name}: "
            f"image={(height, width)}, mask={ground_truth.shape}"
        )
    return image, image_array, image_gray, ground_truth, width, height


"""Native-resolution binary metrics (original empty-target conventions)."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_iou_dice(pred, target):
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if target.sum() == 0:
        return (1.0, 1.0) if pred.sum() == 0 else (0.0, 0.0)
    return inter / union, 2 * inter / (pred.sum() + target.sum())


def compute_auc_metrics(prob_map, gt_mask):
    y_true, y_scores = gt_mask.flatten().astype(int), prob_map.flatten()
    if np.all(y_true == 0) or np.all(y_true == 1):
        return 0.5, 0.5
    return roc_auc_score(y_true, y_scores), average_precision_score(y_true, y_scores)
