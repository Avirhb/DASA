"""Instance-based DASA PBAS. No mutable experiment globals or parent-repo imports.

512 input / 32x32 grid / five prefix tokens are the published ViT-L/16
architecture contract, validated by configuration instead of silently generalized.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
from torchvision import transforms
from .numerics import stable_seed, deterministic_torch_kmeans


class DASA:
    def __init__(self, config, device="cpu", dtype=torch.float32):
        self.config = dict(config)
        self.device = device
        self.dtype = dtype
        if config["input_size"] != 512:
            raise ValueError(
                "Published DINOv3 ViT-L/16 pipeline requires input_size=512"
            )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config["normalization_mean"], std=config["normalization_std"]
                ),
            ]
        )

    def apply_prob_post_processing(self, anomaly_map, img_gray, w, h):
        anomaly_map = (anomaly_map - anomaly_map.min()) / (
            anomaly_map.max() - anomaly_map.min() + 1e-06
        )
        smooth_field = cv2.GaussianBlur(
            cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_CUBIC),
            (
                int(w * self.config["smooth_width_ratio"]) | 1,
                int(w * self.config["smooth_width_ratio"]) | 1,
            ),
            0,
        )
        darkness = np.clip(1.0 - img_gray, 0, None)
        pos_map = np.tile(
            (
                1.0
                / (
                    1.0
                    + np.exp(
                        (np.linspace(0, 1, h) - self.config["pos_map_center"])
                        * self.config["vertical_slope"]
                    )
                )
            )[:, None],
            (1, w),
        )
        p = smooth_field * darkness ** self.config["darkness_exponent"] * pos_map
        return (p - p.min()) / (p.max() - p.min() + 1e-06)

    def get_gaussian_fps_points(self, prob_map, mask_bool, k=8):
        h, w = prob_map.shape
        p = prob_map.copy()
        p[~mask_bool] = 0
        vals = p[p > 0]
        if len(vals) < self.config["min_point_pixels"]:
            return []
        binary = (p >= np.percentile(vals, self.config["point_percentile"])).astype(
            np.uint8
        )
        n, labels, _, _ = cv2.connectedComponentsWithStats(binary)
        if n <= 1:
            return []
        best = np.argmax([np.sum(p[labels == i]) for i in range(1, n)]) + 1
        coords = np.argwhere(labels == best)
        probs = prob_map[labels == best]
        if not len(probs):
            return []
        sel = [np.argmax(probs)]
        p0 = coords[sel[0]]
        sigma = min(h, w) * self.config["point_sigma_ratio"]
        for _ in range(1, k):
            sc = coords[sel]
            d = np.min(
                np.linalg.norm(coords[:, None, :] - sc[None, :, :], axis=2), axis=1
            )
            gp = np.exp(-np.linalg.norm(coords - p0, axis=1) ** 2 / (2 * sigma**2))
            ni = np.argmax(probs * d * gp)
            if (probs * d * gp)[ni] <= 0:
                break
            sel.append(ni)
        return np.array([[coords[i][1], coords[i][0]] for i in sel])

    def component_anchor_bbox(
        self, box_init, prob_map, w, h, core_ratio=0.6, safe_ratio=0.3, pad=10
    ):
        x1, y1, x2, y2 = box_init
        local = prob_map[y1:y2, x1:x2]
        max_val = np.max(local) if local.size > 0 else 0
        mask_safe = cv2.morphologyEx(
            (prob_map > max_val * safe_ratio).astype(np.uint8),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        mask_core = (prob_map > max_val * core_ratio).astype(np.uint8)
        n_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(mask_safe)
        best, max_ov = (0, 0)
        for i in range(1, n_s):
            ov = np.sum(np.logical_and(labels_s == i, mask_core))
            if ov > max_ov:
                max_ov, best = (ov, i)
        if best == 0:
            bm = np.zeros_like(mask_safe)
            bm[y1:y2, x1:x2] = 1
            for i in range(1, n_s):
                ov = np.sum(np.logical_and(labels_s == i, bm))
                if ov > max_ov:
                    max_ov, best = (ov, i)
        if best > 0:
            sx, sy, sw, sh = stats_s[best][:4]
            return np.array(
                [
                    max(0, sx - pad),
                    max(0, sy - pad),
                    min(w, sx + sw + pad),
                    min(h, sy + sh + pad),
                ]
            )
        return np.array([x1, y1, x2, y2])

    def clean_mask(self, mask, k_dir=17, k_smooth=5):
        k_dir = self.config["clean_direction_kernel"]
        k_smooth = self.config["clean_smooth_kernel"]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if n > 1:
            keep = (
                np.where(
                    stats[1:, cv2.CC_STAT_AREA]
                    > np.max(stats[1:, cv2.CC_STAT_AREA])
                    * self.config["component_keep_ratio"]
                )[0]
                + 1
            )
            filtered = np.isin(labels, keep).astype(np.uint8)
        else:
            filtered = mask.astype(np.uint8)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (k_dir, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_dir))
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_smooth, k_smooth))
        combined = cv2.bitwise_or(
            cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kh),
            cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kv),
        )
        filled = ndimage.binary_fill_holes(
            cv2.morphologyEx(combined, cv2.MORPH_CLOSE, ke)
        ).astype(np.uint8)
        n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
            filled, connectivity=8
        )
        if n2 <= 1:
            return filled.astype(bool)
        return labels2 == 1 + np.argmax(stats2[1:, cv2.CC_STAT_AREA])

    @torch.inference_mode()
    def extract_dino_features(self, image, dinov3):
        image_tensor = (
            self.transform(
                image.resize((self.config["input_size"], self.config["input_size"]))
            )
            .unsqueeze(0)
            .to(self.device, dtype=self.dtype)
        )
        output = dinov3(pixel_values=image_tensor, output_hidden_states=False)
        global_feature = F.normalize(output.last_hidden_state[:, 0, :].float(), dim=-1)
        dense_features = F.normalize(
            output.last_hidden_state[:, 5:, :].squeeze(0).float(), dim=-1
        )
        return (global_feature, dense_features)

    def build_support_banks(self, image_dir, mask_dir, support_orders, dinov3):
        unique_supports = sorted(
            {
                image_id
                for entry in support_orders.values()
                for image_id in entry["order"]
            }
        )
        cache = {}
        progress = tqdm(unique_supports, desc="Support feature extraction")
        for image_id in progress:
            with Image.open(image_dir / image_id) as image_handle:
                image = image_handle.convert("RGB")
                global_feature, dense_features = self.extract_dino_features(
                    image, dinov3
                )
            mask = cv2.imread(str(mask_dir / image_id), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(
                    f"Failed to read support mask: {mask_dir / image_id}"
                )
            mask_32 = cv2.resize(mask, (32, 32)) > self.config["mask_threshold"]
            safe_background = (
                cv2.dilate(
                    mask_32.astype(np.uint8),
                    np.ones(
                        (self.config["support_dilation_kernel"],) * 2, dtype=np.uint8
                    ),
                    iterations=1,
                )
                == 0
            )
            cache[image_id] = {
                "global": global_feature,
                "dense": dense_features,
                "mask": mask_32,
                "safe_background": safe_background,
            }
        banks = {}
        for seed, entry in support_orders.items():
            ordered_items = [cache[name] for name in entry["order"]]
            banks[seed] = {
                "globals": torch.cat([item["global"] for item in ordered_items], dim=0),
                "dense_feats": [item["dense"] for item in ordered_items],
                "masks": [item["mask"] for item in ordered_items],
                "bg_safe_masks": [item["safe_background"] for item in ordered_items],
            }
        return banks

    def _upsample_response_map(self, response_map, width, height):
        """Upsample a 32x32 response without image-intensity/spatial priors.

        The canonical DASA map applies the optional darkness and vertical-position
        terms through ``self.apply_prob_post_processing``.  For qualitative
        decomposition figures we also expose the underlying response itself so
        that low-position features are not visually suppressed.
        """
        response_map = np.asarray(response_map, dtype=np.float32)
        response_map = (response_map - response_map.min()) / (
            response_map.max() - response_map.min() + 1e-06
        )
        upsampled = cv2.resize(
            response_map, (width, height), interpolation=cv2.INTER_CUBIC
        )
        kernel_width = max(3, int(width * self.config["smooth_width_ratio"]) | 1)
        smoothed = cv2.GaussianBlur(upsampled, (kernel_width, kernel_width), 0)
        return (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-06)

    def compute_probability_components(
        self, q_global, q_dense, img_gray, width, height, memory_bank, shots, image_id
    ):
        globals_k = memory_bank["globals"][:shots]
        similarities = torch.mm(q_global, globals_k.t()).squeeze(0)
        actual_top_k = min(self.config["top_k"], shots)
        top_indices = torch.topk(similarities, actual_top_k).indices
        positive_features = []
        negative_features = []
        for index in top_indices.cpu().numpy():
            if np.any(memory_bank["masks"][index]):
                positive_features.append(
                    memory_bank["dense_feats"][index][
                        memory_bank["masks"][index].flatten()
                    ]
                )
            if np.any(memory_bank["bg_safe_masks"][index]):
                negative_features.append(
                    memory_bank["dense_feats"][index][
                        memory_bank["bg_safe_masks"][index].flatten()
                    ]
                )
        if positive_features and negative_features:
            foreground_prototype = F.normalize(
                torch.cat(positive_features).mean(0, keepdim=True), dim=-1
            )
            background_features = torch.cat(negative_features)
            background_prototypes = deterministic_torch_kmeans(
                background_features,
                self.config["semantic_bg_k"],
                stable_seed(self.config["kmeans_seed"], image_id, shots),
                n_iter=self.config["kmeans_iterations"],
            )
            semantic_similarity = torch.clamp(
                torch.mm(q_dense, foreground_prototype.t())
                - self.config["contrast_alpha"]
                * torch.mm(q_dense, background_prototypes.t())
                .max(dim=1)[0]
                .view(-1, 1),
                min=0,
            )
            semantic_response = semantic_similarity.view(32, 32).cpu().numpy()
            p_semantic = self.apply_prob_post_processing(
                semantic_response, img_gray, width, height
            )
        else:
            semantic_response = np.ones((32, 32), dtype=np.float32)
            p_semantic = np.ones((height, width), dtype=np.float32)
        p_semantic_raw = self._upsample_response_map(semantic_response, width, height)
        semantic_512 = cv2.resize(p_semantic, (512, 512), interpolation=cv2.INTER_AREA)
        max_value = semantic_512.max()
        anchor_mask = (semantic_512 > max_value * self.config["anchor_ratio"]).astype(
            np.uint8
        )
        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            anchor_mask, connectivity=8
        )
        feature_side = 32
        margin = max(1, int(feature_side * self.config["background_margin_ratio"]))
        margin_top = margin_left = margin_right = margin
        if label_count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_area = np.max(areas)
            largest_label = 1 + np.argmax(areas)
            concentration = largest_area / (np.sum(areas) + 1e-06)
            if (
                concentration > self.config["anchor_concentration"]
                and largest_area > self.config["anchor_min_pixels"]
            ):
                y_indices, x_indices = np.where(labels == largest_label)
                margin_top = min(margin, int(y_indices.min() / 16))
                margin_left = min(margin, int(x_indices.min() / 16))
                margin_right = min(margin, int(31 - x_indices.max() / 16))
        adaptive_mask = torch.zeros(
            (feature_side, feature_side), dtype=torch.bool, device=self.device
        )
        if margin_top > 0:
            adaptive_mask[:margin_top, :] = True
        if margin_left > 0:
            adaptive_mask[:, :margin_left] = True
        if margin_right > 0:
            adaptive_mask[:, feature_side - margin_right :] = True
        if adaptive_mask.sum() == 0:
            adaptive_mask[0, :] = True
            adaptive_mask[:, 0] = True
            adaptive_mask[:, -1] = True
        feature_grid = q_dense.view(feature_side, feature_side, -1)
        query_background = F.normalize(
            feature_grid[adaptive_mask].mean(0, keepdim=True), dim=-1
        )
        background_response = (
            1.0
            - torch.mm(q_dense, query_background.t())
            .view(feature_side, feature_side)
            .cpu()
            .numpy()
        )
        p_background = self.apply_prob_post_processing(
            background_response, img_gray, width, height
        )
        p_background_raw = self._upsample_response_map(
            background_response, width, height
        )
        probability_map = p_background * p_semantic
        p_final = (probability_map - probability_map.min()) / (
            probability_map.max() - probability_map.min() + 1e-06
        )
        return {
            "ssp": np.asarray(p_semantic, dtype=np.float32),
            "ttba": np.asarray(p_background, dtype=np.float32),
            "final": np.asarray(p_final, dtype=np.float32),
            "ssp_raw": np.asarray(p_semantic_raw, dtype=np.float32),
            "ttba_raw": np.asarray(p_background_raw, dtype=np.float32),
            "final_raw": np.asarray(
                p_semantic_raw
                * p_background_raw
                / (np.max(p_semantic_raw * p_background_raw) + 1e-06),
                dtype=np.float32,
            ),
        }

    def compute_probability_map(
        self, q_global, q_dense, img_gray, width, height, memory_bank, shots, image_id
    ):
        """Return the final DASA map while preserving the historical API."""
        return self.compute_probability_components(
            q_global, q_dense, img_gray, width, height, memory_bank, shots, image_id
        )["final"]

    def predict_mask(self, probability_map, sam, width, height):
        max_value = probability_map.max()
        candidate_mask = (
            probability_map > max_value * self.config["candidate_ratio"]
        ).astype(np.uint8)
        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask)
        candidates = [
            index
            for index in range(1, label_count)
            if stats[index, cv2.CC_STAT_AREA] > self.config["candidate_min_pixels"]
        ] or [0]
        best_score = -999.0
        best_label = candidates[0]
        for candidate_label in candidates:
            if candidate_label == 0:
                candidate_region = np.ones((height, width), dtype=bool)
            else:
                candidate_region = labels == candidate_label
            values = probability_map[candidate_region]
            if len(values):
                core_values = values[
                    values >= values.max() * self.config["candidate_core_ratio"]
                ]
                score = np.mean(core_values) if len(core_values) else 0.0
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_label = candidate_label
        if best_label == 0:
            selected_region = np.ones((height, width), dtype=bool)
            initial_box = np.array([0, 0, width, height])
        else:
            selected_region = labels == best_label
            x_pos, y_pos, box_width, box_height = stats[best_label][:4]
            initial_box = np.array(
                [x_pos, y_pos, x_pos + box_width, y_pos + box_height]
            )
        final_box = self.component_anchor_bbox(
            initial_box,
            probability_map,
            width,
            height,
            core_ratio=self.config["bbox_core_ratio"],
            safe_ratio=self.config["bbox_safe_ratio"],
            pad=self.config["bbox_padding"],
        )
        points = self.get_gaussian_fps_points(
            probability_map, selected_region, k=self.config["num_point_prompts"]
        )
        if len(points):
            masks, _, _ = sam.predict(
                point_coords=points,
                point_labels=np.ones(len(points)),
                box=final_box,
                multimask_output=False,
            )
        else:
            masks, _, _ = sam.predict(box=final_box, multimask_output=False)
        return self.clean_mask(masks[0])
