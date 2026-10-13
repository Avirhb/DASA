"""Small JSON/CSV and path helpers."""

import csv
import json
import os
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve(value, base):
    return (Path(base) / os.path.expandvars(os.path.expanduser(value))).resolve()


def load_config(filename, paths_file=None):
    filename = Path(filename).resolve()
    config = read_json(filename)
    paths_file = (
        Path(paths_file).resolve()
        if paths_file
        else resolve(config["paths_config"], filename.parent)
    )
    paths = read_json(paths_file)
    for key in (
        "data_root",
        "dinov3_model",
        "sam_checkpoint",
        "abd_root",
        "swin_checkpoint",
    ):
        paths[key] = str(resolve(paths[key], paths_file.parent))
    paths["asset_roots"] = {
        k: str(resolve(v, paths_file.parent))
        for k, v in paths.get("asset_roots", {}).items()
    }
    for dirs in paths.get("dataset_directories", {}).values():
        for key in dirs:
            dirs[key] = str(resolve(dirs[key], paths_file.parent))
    for key in ("output_dir", "manifest_dir", "swin_config"):
        if key in config:
            config[key] = str(resolve(config[key], filename.parent))
    return config, paths


def resolve_asset(value, roots):
    if "://" not in value:
        return Path(value)
    alias, relative = value.split("://", 1)
    root = Path(roots[alias]).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Asset escapes root: {value}")
    return path


def load_manifests(folder):
    folder = Path(folder).resolve()
    roles = {
        role: read_csv(folder / f"{role}_manifest.csv")
        for role in ("labeled", "unlabeled", "test")
    }
    if any(not rows for rows in roles.values()):
        raise ValueError("Labeled, unlabeled and test manifests must be nonempty")
    for rows in roles.values():
        for row in rows:
            for key in ("image_path", "mask_path"):
                if row[key] and "://" not in row[key]:
                    row[key] = str(resolve(row[key], folder))
    if any(r["mask_path"] for r in roles["unlabeled"]):
        raise ValueError("Unlabeled rows must have blank mask_path")
    if any(not r["mask_path"] for r in roles["labeled"] + roles["test"]):
        raise ValueError("Support, accepted labels and test cases require masks")
    for field in ("case_id", "source_group_id"):
        train = {r[field] for r in roles["labeled"] + roles["unlabeled"]}
        test = {r[field] for r in roles["test"]}
        if train & test:
            raise ValueError(f"Training/test overlap in {field}")
    labeled = roles["labeled"]
    if sum(r["label_source"] == "one_shot_support_gt" for r in labeled) != 1:
        raise ValueError("Expected one ground-truth support")
    for row in labeled:
        if row["label_source"] != "one_shot_support_gt":
            if (
                row["label_source"] != "frozen_dasa"
                or not 1 <= int(row["accept_rank"]) <= 30
            ):
                raise ValueError(
                    "Other labels must come from the first 30 real accepts"
                )
    return roles
