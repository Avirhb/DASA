"""Optional IGFS branch preserving original FP16 support / FP32 query features.

Its sequential NumPy sampling stream includes the support and completed queries.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans
from .dasa import DASA


class IGFS(DASA):
    @torch.inference_mode()
    def extract_reference_dino_features(self, image, dinov3, support=False):
        image_tensor = (
            self.transform(
                image.resize((self.config["input_size"], self.config["input_size"]))
            )
            .unsqueeze(0)
            .to(self.device)
        )
        if support and str(self.device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = dinov3(pixel_values=image_tensor, output_hidden_states=True)
        else:
            output = dinov3(pixel_values=image_tensor, output_hidden_states=True)
        global_feature = F.normalize(output.last_hidden_state[:, 0, :], dim=-1)
        dense_features = F.normalize(
            output.hidden_states[self.config["feature_layer_idx"] + 1][
                :, 5:, :
            ].squeeze(0),
            dim=-1,
        )
        return (global_feature, dense_features)

    def apply_igfs_post_processing(
        self, anomaly_map, image_gray, width, height, pos_map_center=None
    ):
        pos_map_center = (
            self.config["pos_map_center"] if pos_map_center is None else pos_map_center
        )
        anomaly_map = (anomaly_map - anomaly_map.min()) / (
            anomaly_map.max() - anomaly_map.min() + 1e-06
        )
        smooth_field = cv2.resize(
            anomaly_map, (width, height), interpolation=cv2.INTER_CUBIC
        )
        kernel_size = int(width * self.config["smooth_width_ratio"]) | 1
        smooth_field = cv2.GaussianBlur(smooth_field, (kernel_size, kernel_size), 0)
        darkness = np.clip(1.0 - image_gray, 0, None)
        positions = np.linspace(0, 1, height)
        position_prior = 1.0 / (
            1.0 + np.exp((positions - pos_map_center) * self.config["vertical_slope"])
        )
        position_map = np.tile(position_prior[:, None], (1, width))
        probability = (
            smooth_field * darkness ** self.config["darkness_exponent"] * position_map
        )
        return (probability - probability.min()) / (
            probability.max() - probability.min() + 1e-06
        )

    def build_support_bank(self, image_dir, mask_dir, support_image, dinov3):
        with Image.open(image_dir / support_image) as image_handle:
            support_image_pil = image_handle.convert("RGB")
        support_global, support_dense = self.extract_reference_dino_features(
            support_image_pil, dinov3, support=True
        )
        support_mask = cv2.imread(str(mask_dir / support_image), cv2.IMREAD_GRAYSCALE)
        if support_mask is None:
            raise RuntimeError(
                f"Failed to read support mask: {mask_dir / support_image}"
            )
        mask32 = cv2.resize(support_mask, (32, 32)) > self.config["mask_threshold"]
        safe_background = (
            cv2.dilate(
                mask32.astype(np.uint8),
                np.ones((self.config["support_dilation_kernel"],) * 2, dtype=np.uint8),
                iterations=1,
            )
            == 0
        )
        positive = support_dense[mask32.flatten()]
        negative = support_dense[safe_background.flatten()]
        if not len(positive) or not len(negative):
            raise RuntimeError(
                f"Support image has an empty foreground/background feature set: {support_image}"
            )
        foreground_prototype = F.normalize(positive.mean(0, keepdim=True), dim=-1)
        centers = (
            KMeans(
                n_clusters=self.config["semantic_bg_k"],
                random_state=self.config["igfs_kmeans_seed"],
                n_init="auto",
            )
            .fit(negative.detach().cpu().numpy())
            .cluster_centers_
        )
        background_prototypes = F.normalize(
            torch.tensor(centers).to(self.device), dim=-1
        )
        return {
            "global": support_global,
            "dense": support_dense,
            "mask32": mask32,
            "safe_background32": safe_background,
            "foreground_prototype": foreground_prototype,
            "background_prototypes": background_prototypes,
        }

    def compute_igfs_probability(
        self,
        q_dense,
        image_gray,
        width,
        height,
        support_bank,
        sample_ratio,
        k_sigma,
        dataset,
        image_id,
        pos_map_center,
        sampling_rng=None,
    ):
        semantic_similarity = torch.clamp(
            torch.mm(q_dense, support_bank["foreground_prototype"].t())
            - self.config["contrast_alpha"]
            * torch.mm(q_dense, support_bank["background_prototypes"].t())
            .max(dim=1)[0]
            .view(-1, 1),
            min=0,
        ).view(32, 32)
        p_sem = self.apply_igfs_post_processing(
            semantic_similarity.cpu().numpy(), image_gray, width, height, pos_map_center
        )
        p_sem_32 = cv2.resize(p_sem, (32, 32), interpolation=cv2.INTER_AREA)
        threshold = np.percentile(p_sem_32, self.config["point_percentile"])
        mask_high_prob = (p_sem_32 >= threshold).astype(np.uint8)
        label_count, _, _, centroids = cv2.connectedComponentsWithStats(
            mask_high_prob, connectivity=8
        )
        centroid_mask = np.zeros((32, 32), dtype=bool)
        if label_count > 1:
            for index in range(1, label_count):
                x_pos, y_pos = centroids[index]
                x_pos = int(np.clip(x_pos, 0, 32 - 1))
                y_pos = int(np.clip(y_pos, 0, 32 - 1))
                centroid_mask[y_pos, x_pos] = True
        else:
            centroid_mask[15, 15] = True
        equivalent_radius = np.sqrt(mask_high_prob.sum() / np.pi)
        adaptive_sigma = max(self.config["igfs_min_sigma"], k_sigma * equivalent_radius)
        distance_map = ndimage.distance_transform_edt(~centroid_mask)
        foveated_distance = np.exp(-(distance_map**2) / (2 * adaptive_sigma**2))
        anti_foveated_distance = 1.0 - foveated_distance
        sample_probabilities = anti_foveated_distance / (
            anti_foveated_distance.sum() + 1e-08
        )
        sample_count = int(1024 * sample_ratio)
        if sampling_rng is None:
            sampling_rng = np.random
        sampled_indices = sampling_rng.choice(
            1024, size=sample_count, replace=False, p=sample_probabilities.reshape(-1)
        )
        sampled_mask = torch.zeros((32, 32), dtype=torch.bool, device=self.device)
        sampled_mask.view(-1)[torch.as_tensor(sampled_indices, device=self.device)] = (
            True
        )
        feature_grid = q_dense.view(32, 32, -1)
        query_background = F.normalize(
            feature_grid[sampled_mask].mean(0, keepdim=True), dim=-1
        )
        p_background = self.apply_igfs_post_processing(
            1.0 - torch.mm(q_dense, query_background.t()).view(32, 32).cpu().numpy(),
            image_gray,
            width,
            height,
            pos_map_center,
        )
        probability_map = p_background * p_sem
        probability_map = (probability_map - probability_map.min()) / (
            probability_map.max() - probability_map.min() + 1e-06
        )
        visualization = {
            "anti_foveated_distance": anti_foveated_distance,
            "centroid_mask": centroid_mask,
            "sampled_mask": sampled_mask.cpu().numpy(),
            "p_background": p_background,
            "p_semantic": p_sem,
            "adaptive_sigma": adaptive_sigma,
        }
        return (probability_map, visualization)

    def predict_igfs_mask(self, probability_map, sam, width, height, prompt):
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
        for candidate in candidates:
            region = (
                np.ones((height, width), dtype=bool)
                if candidate == 0
                else labels == candidate
            )
            values = probability_map[region]
            if len(values):
                core_values = values[
                    values >= values.max() * self.config["candidate_core_ratio"]
                ]
                score = float(core_values.mean()) if len(core_values) else 0.0
            else:
                score = 0.0
            if score > best_score:
                best_score, best_label = (score, candidate)
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
            core_ratio=float(prompt["core_ratio"]),
            safe_ratio=float(prompt["safe_ratio"]),
            pad=int(prompt["padding"]),
        )
        points = self.get_gaussian_fps_points(
            probability_map, selected_region, k=int(prompt["num_points"])
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
        return (self.clean_mask(masks[0]), final_box, points)
