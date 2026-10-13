# DASA

**Training-free ultrasound segmentation for one-shot-initiated task learning**

DASA segments ultrasound images from one annotated reference using frozen DINOv3
and SAM. The resulting masks can be reviewed and used for one-shot-initiated task
learning (OITL).

![Study overview](assets/overview.png)

## Setup

Use Python 3.11 with CUDA for GPU runs.

```bash
pip install -r requirements.txt
```

Prepare your datasets and DINOv3 ViT-L/16, SAM ViT-H, and Swin-T weights, then edit
`configs/paths.json`. Paths are relative to the config file. For evaluation, place
images and same-name masks in `data/processed_datasets/<DATASET>/{images,masks}/`.

## One-shot segmentation

```bash
python infer.py --dataset HC18
python infer.py --config configs/igfs.json --dataset HC18
```

These run the PBAS (default) and IGFS variants. Each run selects one reference and
evaluates the remaining images. Settings are in `configs/`; predictions and
metrics are saved under `outputs/`. Use a new output directory for each run.

## OITL

Set up [ABD](https://github.com/Star-chy/ABD/tree/4efd525e052c6e3601c29c1abaa4648de81cf206)
and set `abd_root` in `configs/paths.json`. Set `manifest_dir` in
`configs/task_learning.json` to your `labeled_manifest.csv`,
`unlabeled_manifest.csv`, and `test_manifest.csv` folder.

```bash
python train.py
```

Manifest columns: `image_id`, `case_id`, `source_group_id`, `image_path`,
`mask_path`, `label_source`, `accept_rank`. Use `one_shot_support_gt` for the
reference and `frozen_dasa` for accepted masks; leave unlabeled mask paths empty.

## License

The core implementation is [Apache-2.0](LICENSE-CODE). Previously released
repository material retains its [MIT license](LICENSE); bundled SAM retains its
[original license](third_party/segment_anything/LICENSE).

Contact: chenfang_bme@163.com
