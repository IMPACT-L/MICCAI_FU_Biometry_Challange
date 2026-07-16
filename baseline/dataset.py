import glob
import json
import os
import random
from typing import Iterator, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from utils import canonicalize_task_coords, letterbox_image_and_points


EXTRA_REGRESSION_TASK_IDS = {"A4C", "AOP", "FA", "HC", "IVC", "PLAX", "PSAX"}
CARDIAC_SPLIT_SCREEN_TASK_IDS = {"A4C", "PSAX", "PLAX", "IVC"}
FETAL_FEMUR_ORIENTATION_ANOMALY_BASENAMES = {
    "Patient00757_Plane5_1_of_1.png",
    "Patient00863_Plane5_1_of_2.png",
    "Patient00863_Plane5_2_of_2.png",
    "Patient01025_Plane5_1_of_1.png",
    "Patient01035_Plane5_2_of_4.png",
    "Patient01035_Plane5_4_of_4.png",
    "Patient01130_Plane5_2_of_4.png",
    "Patient01221_Plane5_2_of_2.png",
    "Patient01246_Plane5_1_of_2.png",
    "Patient01248_Plane5_1_of_1.png",
    "Patient01249_Plane5_1_of_1.png",
    "Patient01301_Plane5_1_of_2.png",
    "Patient01301_Plane5_2_of_2.png",
    "Patient01304_Plane5_2_of_2.png",
    "Patient01475_Plane5_1_of_1.png",
    "Patient01476_Plane5_1_of_1.png",
    "Patient01477_Plane5_1_of_2.png",
    "Patient01478_Plane5_1_of_1.png",
    "Patient01480_Plane5_1_of_1.png",
    "Patient01481_Plane5_1_of_1.png",
    "Patient01605_Plane5_2_of_2.png",
    "Patient01606_Plane5_2_of_2.png",
    "Patient01607_Plane5_1_of_2.png",
    "Patient01608_Plane5_1_of_1.png",
    "Patient01609_Plane5_1_of_1.png",
}


class KeypointDataset(Dataset):
    """Dataset for keypoint localization tasks only."""

    def __init__(
        self,
        data_root: str,
        transforms: Optional[A.Compose] = None,
        heatmap_size: Tuple[int, int] = (64, 64),
        sigma: float = 1.8,
        input_size: int = 518,
        roi_crop: bool = False,
        roi_crop_tasks: Optional[set[str]] = None,
        roi_context_range: Tuple[float, float] = (1.4, 2.0),
        roi_center_jitter: float = 0.0,
        roi_anchor_json: Optional[str] = None,
        cardiac_split_screen_mode: str = "keep",
        exclude_fetal_femur_orientation_anomalies: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.input_size = input_size
        self.roi_crop = roi_crop
        self.roi_crop_tasks = roi_crop_tasks
        self.roi_context_range = roi_context_range
        self.roi_center_jitter = float(roi_center_jitter)
        self.roi_anchor_predictions = self._load_roi_anchor_predictions(roi_anchor_json)
        self.cardiac_split_screen_mode = str(cardiac_split_screen_mode)
        self.exclude_fetal_femur_orientation_anomalies = bool(
            exclude_fetal_femur_orientation_anomalies
        )
        self.csv_path = os.path.join(self.data_root, "csv")
        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")

        all_csv_files = glob.glob(os.path.join(self.csv_path, "*.csv"))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")

        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)

        is_regression = dataframe["task_name"].astype(str).eq("Regression")
        is_extra_task = dataframe["task_id"].astype(str).isin(EXTRA_REGRESSION_TASK_IDS)
        self.dataframe = dataframe[is_regression | is_extra_task].reset_index(drop=True)
        if self.dataframe.empty:
            raise ValueError(
                "No keypoint records found. Expect task_name == 'Regression' or task_id in "
                f"{sorted(EXTRA_REGRESSION_TASK_IDS)}."
            )

        if self.exclude_fetal_femur_orientation_anomalies:
            anomaly_mask = self._fetal_femur_orientation_anomaly_mask(self.dataframe)
            anomaly_count = int(anomaly_mask.sum())
            if anomaly_count:
                self.dataframe = self.dataframe[~anomaly_mask].reset_index(drop=True)
                print(
                    "Filtered out "
                    f"{anomaly_count} fetal_femur orientation-anomaly samples "
                    "listed by the challenge organizers."
                )

        valid_indices = []
        missing_paths = []
        for idx, image_path in enumerate(self.dataframe["image_path"].astype(str).tolist()):
            resolved = self._resolve_image_path(image_path)
            if resolved is None:
                missing_paths.append(image_path)
            else:
                valid_indices.append(idx)

        if missing_paths:
            self.dataframe = self.dataframe.iloc[valid_indices].reset_index(drop=True)
            preview = ", ".join(missing_paths[:5])
            print(
                f"Filtered out {len(missing_paths)} samples with missing images. "
                f"Examples: {preview}"
            )
            if self.dataframe.empty:
                raise FileNotFoundError(
                    "All dataset rows were filtered out because their images could not be resolved."
                )

        print(f"Keypoint data loaded. Total samples: {len(self.dataframe)}")

    @staticmethod
    def _normalize_anchor_key(task_id: str, image_path: str) -> tuple[str, str]:
        normalized = os.path.normpath(str(image_path)).replace("\\", "/")
        parts = [part for part in normalized.split("/") if part not in ("", ".")]
        if parts and parts[0] == "images":
            parts = parts[1:]
        if parts and parts[0] == str(task_id):
            return str(task_id), "/".join(parts)
        return str(task_id), f"{task_id}/{os.path.basename(normalized)}"

    @classmethod
    def _load_roi_anchor_predictions(
        cls,
        roi_anchor_json: Optional[str],
    ) -> dict[tuple[str, str], np.ndarray]:
        if not roi_anchor_json:
            return {}
        if not os.path.isfile(roi_anchor_json):
            raise FileNotFoundError(f"ROI anchor JSON not found: {roi_anchor_json}")
        with open(roi_anchor_json, "r", encoding="utf-8") as handle:
            predictions = json.load(handle)
        anchors: dict[tuple[str, str], np.ndarray] = {}
        for item in predictions:
            task_id = str(item["task_id"])
            coords = np.asarray(item["predicted_points_normalized"], dtype=np.float32)
            coords = canonicalize_task_coords(coords, task_id).reshape(-1, 2)
            key = cls._normalize_anchor_key(task_id, str(item["image_path"]))
            anchors[key] = coords
            anchors[(task_id, os.path.basename(str(item["image_path"])))] = coords
        print(f"Loaded ROI anchor predictions: {len(predictions)} rows from {roi_anchor_json}")
        return anchors

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def _fetal_femur_orientation_anomaly_mask(dataframe: pd.DataFrame) -> pd.Series:
        task_ids = dataframe["task_id"].astype(str)
        basenames = dataframe["image_path"].astype(str).map(os.path.basename)
        return task_ids.eq("fetal_femur") & basenames.isin(
            FETAL_FEMUR_ORIENTATION_ANOMALY_BASENAMES
        )

    def _resolve_image_path(self, rel_path: str) -> Optional[str]:
        rel_norm = os.path.normpath(rel_path)
        cleaned_rel = rel_norm
        while cleaned_rel.startswith(".." + os.sep):
            cleaned_rel = cleaned_rel[3:]

        for root in [os.path.join(self.data_root, "images"), self.data_root]:
            direct = os.path.normpath(os.path.join(root, cleaned_rel))
            if os.path.isfile(direct):
                return direct

        return None

    def _generate_heatmaps(self, norm_coords: np.ndarray, num_points: int) -> np.ndarray:
        heatmap_h, heatmap_w = self.heatmap_size
        yy, xx = np.meshgrid(np.arange(heatmap_h), np.arange(heatmap_w), indexing="ij")
        heatmaps = np.zeros((num_points, heatmap_h, heatmap_w), dtype=np.float32)

        for i in range(num_points):
            x_norm = float(norm_coords[2 * i])
            y_norm = float(norm_coords[2 * i + 1])

            x_norm = min(max(x_norm, 0.0), 1.0)
            y_norm = min(max(y_norm, 0.0), 1.0)

            x = x_norm * (heatmap_w - 1)
            y = y_norm * (heatmap_h - 1)

            dist2 = (xx - x) ** 2 + (yy - y) ** 2
            heatmaps[i] = np.exp(-dist2 / (2.0 * self.sigma * self.sigma)).astype(np.float32)

        return heatmaps

    def _should_apply_roi_crop(self, task_id: str) -> bool:
        if not self.roi_crop:
            return False
        if self.roi_crop_tasks is None:
            return True
        return str(task_id) in self.roi_crop_tasks

    def _lookup_roi_anchor_points(
        self,
        task_id: str,
        image_path: str,
        image_width: int,
        image_height: int,
        expected_points: int,
    ) -> Optional[np.ndarray]:
        if not self.roi_anchor_predictions:
            return None
        key = self._normalize_anchor_key(task_id, image_path)
        anchor = self.roi_anchor_predictions.get(key)
        if anchor is None:
            anchor = self.roi_anchor_predictions.get((str(task_id), os.path.basename(str(image_path))))
        if anchor is None or len(anchor) != expected_points:
            return None
        anchor_px = anchor.copy()
        anchor_px[:, 0] *= max(float(image_width - 1), 1.0)
        anchor_px[:, 1] *= max(float(image_height - 1), 1.0)
        return anchor_px.astype(np.float32)

    def _apply_roi_crop(
        self,
        image: np.ndarray,
        coords_px: np.ndarray,
        reference_coords_px: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        crop_reference = reference_coords_px if reference_coords_px is not None else coords_px
        valid_coords = crop_reference[np.isfinite(crop_reference).all(axis=1)]
        if len(valid_coords) == 0:
            return image, coords_px, reference_coords_px

        image_h, image_w = image.shape[:2]
        x_min = float(valid_coords[:, 0].min())
        y_min = float(valid_coords[:, 1].min())
        x_max = float(valid_coords[:, 0].max())
        y_max = float(valid_coords[:, 1].max())

        box_w = max(x_max - x_min, 8.0)
        box_h = max(y_max - y_min, 8.0)
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)

        context_min, context_max = self.roi_context_range
        context = float(np.random.uniform(context_min, context_max))
        crop_w = min(float(image_w), max(box_w * context, box_w + 32.0))
        crop_h = min(float(image_h), max(box_h * context, box_h + 32.0))
        if self.roi_center_jitter > 0.0:
            center_x += float(np.random.uniform(-self.roi_center_jitter, self.roi_center_jitter)) * crop_w
            center_y += float(np.random.uniform(-self.roi_center_jitter, self.roi_center_jitter)) * crop_h

        x0 = int(np.floor(np.clip(center_x - crop_w * 0.5, 0.0, max(image_w - crop_w, 0.0))))
        y0 = int(np.floor(np.clip(center_y - crop_h * 0.5, 0.0, max(image_h - crop_h, 0.0))))
        x1 = int(np.ceil(min(float(image_w), x0 + crop_w)))
        y1 = int(np.ceil(min(float(image_h), y0 + crop_h)))

        cropped_image = image[y0:y1, x0:x1]
        cropped_coords = coords_px.copy()
        cropped_coords[:, 0] -= float(x0)
        cropped_coords[:, 1] -= float(y0)
        cropped_reference = None
        if reference_coords_px is not None:
            cropped_reference = reference_coords_px.copy()
            cropped_reference[:, 0] -= float(x0)
            cropped_reference[:, 1] -= float(y0)
        return cropped_image, cropped_coords, cropped_reference

    def _should_crop_cardiac_split_screen(self, record, task_id: str) -> bool:
        if self.cardiac_split_screen_mode != "crop_panel":
            return False
        if str(task_id) not in CARDIAC_SPLIT_SCREEN_TASK_IDS:
            return False
        if "is_split_screen_cardiac" not in record:
            return False
        return bool(record["is_split_screen_cardiac"])

    @staticmethod
    def _find_split_screen_seam_x(image: np.ndarray) -> int:
        image_h, image_w = image.shape[:2]
        if image_w < 8 or image_h < 8:
            return image_w // 2
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        band_left = int(round(0.42 * image_w))
        band_right = int(round(0.58 * image_w))
        band_left = max(1, min(band_left, image_w - 2))
        band_right = max(band_left + 1, min(band_right, image_w - 1))
        central_profile = gray[:, band_left:band_right].mean(axis=0)
        return int(band_left + int(np.argmin(central_profile)))

    def _apply_cardiac_split_screen_panel_crop(
        self,
        image: np.ndarray,
        coords_px: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid_coords = coords_px[np.isfinite(coords_px).all(axis=1)]
        if len(valid_coords) == 0:
            return image, coords_px

        image_h, image_w = image.shape[:2]
        if image_w < 64 or image_h < 64:
            return image, coords_px

        seam_x = self._find_split_screen_seam_x(image)
        median_x = float(np.median(valid_coords[:, 0]))
        if median_x <= float(seam_x):
            x0 = 0
            x1 = max(seam_x - 2, 1)
        else:
            x0 = min(seam_x + 2, image_w - 1)
            x1 = image_w

        crop_w = x1 - x0
        if crop_w < int(0.35 * image_w):
            return image, coords_px

        shifted = coords_px.copy()
        shifted[:, 0] -= float(x0)
        finite_shifted = shifted[np.isfinite(shifted).all(axis=1)]
        if len(finite_shifted) and (
            float(finite_shifted[:, 0].min()) < -1.0
            or float(finite_shifted[:, 0].max()) > float(crop_w)
        ):
            return image, coords_px

        return image[:, x0:x1], shifted

    def _apply_transforms(
        self,
        image: np.ndarray,
        coords_px: np.ndarray,
    ) -> tuple[torch.Tensor | np.ndarray, np.ndarray]:
        if self.transforms is None:
            return image, coords_px

        has_keypoint_support = bool(
            getattr(self.transforms, "processors", None)
            and "keypoints" in self.transforms.processors
        )
        if not has_keypoint_support:
            augmented = self.transforms(image=image)
            return augmented["image"], coords_px

        keypoints = [tuple(map(float, point)) for point in coords_px.tolist()]
        augmented = self.transforms(image=image, keypoints=keypoints)
        transformed_points = np.array(augmented["keypoints"], dtype=np.float32).reshape(-1, 2)
        return augmented["image"], transformed_points

    def __getitem__(self, idx: int) -> dict:
        total = len(self)
        if total == 0:
            raise IndexError("KeypointDataset is empty.")

        record = None
        image = None
        task_id = None
        last_error = None
        for offset in range(total):
            current_idx = (idx + offset) % total
            record = self.dataframe.iloc[current_idx]
            task_id = record["task_id"]
            image_abs_path = self._resolve_image_path(record["image_path"])
            if image_abs_path is None:
                last_error = f"Could not resolve image path: {record['image_path']}"
                continue
            image = cv2.imread(image_abs_path)
            if image is None:
                last_error = f"Failed to read image: {image_abs_path}"
                continue
            break
        else:
            raise FileNotFoundError(
                "Unable to load any valid image from dataset sample search. "
                f"Last error: {last_error}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        num_points = int(record["num_classes"])
        coords = []
        for i in range(1, num_points + 1):
            col = f"point_{i}_xy"
            if col in record and pd.notna(record[col]):
                coords.extend(json.loads(record[col]))
            else:
                coords.extend([0.0, 0.0])
        label = np.array(coords, dtype=np.float32)
        label = canonicalize_task_coords(label, task_id)
        label_points = label.reshape(-1, 2)

        if self._should_crop_cardiac_split_screen(record, task_id):
            image, label_points = self._apply_cardiac_split_screen_panel_crop(image, label_points)

        anchor_points = None
        if self._should_apply_roi_crop(task_id):
            image_h, image_w = image.shape[:2]
            anchor_points = self._lookup_roi_anchor_points(
                str(task_id),
                str(record["image_path"]),
                image_width=image_w,
                image_height=image_h,
                expected_points=num_points,
            )
            image, label_points, anchor_points = self._apply_roi_crop(
                image,
                label_points,
                reference_coords_px=anchor_points,
            )

        original_height, original_width = image.shape[:2]
        label_original = label_points.reshape(-1).astype(np.float32)

        image, transformed_points, meta = letterbox_image_and_points(
            image=image,
            coords_px=label_points,
            input_size=self.input_size,
        )
        transformed_anchor = None
        if anchor_points is not None:
            _, transformed_anchor, _ = letterbox_image_and_points(
                image=np.zeros((original_height, original_width, 3), dtype=np.uint8),
                coords_px=anchor_points,
                input_size=self.input_size,
            )
            combined_points = np.concatenate([transformed_points, transformed_anchor], axis=0)
            image, combined_points = self._apply_transforms(image, combined_points)
            transformed_points = combined_points[:num_points]
            transformed_anchor = combined_points[num_points:]
        else:
            image, transformed_points = self._apply_transforms(image, transformed_points)
        transformed_label = transformed_points.reshape(-1).astype(np.float32)
        anchor_train_label = (
            transformed_anchor.reshape(-1).astype(np.float32)
            if transformed_anchor is not None
            else transformed_label.copy()
        )

        transformed_label[0::2] /= max(float(self.input_size - 1), 1.0)
        transformed_label[1::2] /= max(float(self.input_size - 1), 1.0)
        transformed_label = np.clip(transformed_label, 0.0, 1.0)
        anchor_train_label[0::2] /= max(float(self.input_size - 1), 1.0)
        anchor_train_label[1::2] /= max(float(self.input_size - 1), 1.0)
        anchor_train_label = np.clip(anchor_train_label, 0.0, 1.0)

        label_original[0::2] /= max(float(original_width - 1), 1.0)
        label_original[1::2] /= max(float(original_height - 1), 1.0)
        label_original = np.clip(label_original, 0.0, 1.0)

        heatmaps = self._generate_heatmaps(transformed_label, num_points)
        final_label = torch.from_numpy(label_original).float()
        final_heatmaps = torch.from_numpy(heatmaps).float()

        return {
            "image": image,
            "label": final_label,
            "train_label": torch.from_numpy(transformed_label).float(),
            "anchor_train_label": torch.from_numpy(anchor_train_label).float(),
            "has_anchor_label": torch.tensor(transformed_anchor is not None, dtype=torch.bool),
            "heatmap": final_heatmaps,
            "task_id": task_id,
            "meta": meta,
            "image_path": str(record["image_path"]),
            "pseudo_domain": str(record["pseudo_domain"]) if "pseudo_domain" in record else "unknown",
        }


class KeypointUniformSampler(Sampler[List[int]]):
    """Uniform task sampler for keypoint subtasks."""

    def __init__(
        self,
        dataset: KeypointDataset,
        batch_size: int,
        steps_per_epoch: Optional[int] = None,
        task_sampling_weights: Optional[dict[str, float]] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_by_task = {}
        self.task_sampling_weights = task_sampling_weights or {}

        print("\n--- Initializing Keypoint Sampler ---")
        for idx, task_id in enumerate(tqdm(dataset.dataframe["task_id"], desc="Grouping indices")):
            if task_id not in self.indices_by_task:
                self.indices_by_task[task_id] = []
            self.indices_by_task[task_id].append(idx)

        self.task_ids = list(self.indices_by_task.keys())
        self.task_probabilities = [
            max(float(self.task_sampling_weights.get(task_id, 1.0)), 1e-8) for task_id in self.task_ids
        ]
        for task_id in self.task_ids:
            random.shuffle(self.indices_by_task[task_id])

        if steps_per_epoch is None:
            self.steps_per_epoch = len(self.dataset) // self.batch_size
        else:
            self.steps_per_epoch = steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        task_cursors = {task_id: 0 for task_id in self.task_ids}

        for _ in range(self.steps_per_epoch):
            task_id = random.choices(self.task_ids, weights=self.task_probabilities, k=1)[0]
            indices = self.indices_by_task[task_id]
            cursor = task_cursors[task_id]

            start_idx = cursor
            end_idx = start_idx + self.batch_size

            if end_idx > len(indices):
                batch_indices = indices[start_idx:]
                random.shuffle(indices)
                remaining = self.batch_size - len(batch_indices)
                batch_indices.extend(indices[:remaining])
                task_cursors[task_id] = remaining
            else:
                batch_indices = indices[start_idx:end_idx]
                task_cursors[task_id] = end_idx

            yield batch_indices

    def __len__(self) -> int:
        return self.steps_per_epoch
