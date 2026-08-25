import csv
import json
import os
import subprocess
import sys
from pathlib import Path


class Model:
    def __init__(self):
        self.app_dir = Path("/app")

    def _build_manifest(self, data_root: str) -> tuple[Path, list[dict[str, str]]]:
        data_root_path = Path(data_root)
        metadata_path = data_root_path / "csv" / "test_metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing test metadata: {metadata_path}")

        rows: list[dict[str, str]] = []
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "task_id", "num_classes", "height", "width"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"test_metadata.csv missing columns: {sorted(missing)}")
            for row in reader:
                rows.append(row)

        manifest_path = data_root_path / "csv" / "docker_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["task_id", "image_path", "abs_path", "height", "width", "num_points"],
            )
            writer.writeheader()
            for row in rows:
                image_path = row["image_path"]
                writer.writerow(
                    {
                        "task_id": row["task_id"],
                        "image_path": image_path,
                        "abs_path": f"../images/{image_path}",
                        "height": row["height"],
                        "width": row["width"],
                        "num_points": row["num_classes"],
                    }
                )
        return manifest_path, rows

    @staticmethod
    def _write_required_output(output_dir: str, metadata_rows: list[dict[str, str]]) -> None:
        output_path = Path(output_dir) / "regression_predictions.json"
        with output_path.open("r", encoding="utf-8") as handle:
            predictions = json.load(handle)

        by_key = {(item["task_id"], item["image_path"]): item for item in predictions}
        required_only = []
        for row in metadata_rows:
            key = (row["task_id"], row["image_path"])
            if key not in by_key:
                raise KeyError(f"Missing prediction for {key}")
            item = by_key[key]
            required_only.append(
                {
                    "image_path": item["image_path"],
                    "task_id": item["task_id"],
                    "predicted_points_pixels": item["predicted_points_pixels"],
                }
            )

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(required_only, handle, indent=2)

    def predict(self, data_root: str, output_dir: str, batch_size: int = 8):
        # The organizer calls this method with batch_size=8. Higher-resolution
        # checkpoints can exceed the evaluation GPU's memory at that value, so
        # allow an image-specific, baked-in safety cap.
        batch_size = min(batch_size, int(os.environ.get("FU_BIOMETRY_MAX_BATCH_SIZE", batch_size)))
        manifest_path, metadata_rows = self._build_manifest(data_root)
        checkpoint_path = Path(data_root) / "best_model.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")

        cmd = [
            sys.executable,
            str(self.app_dir / "submit.py"),
            "--manifest",
            str(manifest_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            output_dir,
            "--batch-size",
            str(batch_size),
            "--num-workers",
            os.environ.get("NUM_WORKERS", "2"),
            "--encoder-weights",
            "none",
            "--skip-count-validation",
        ]
        subprocess.run(cmd, check=True)
        self._write_required_output(output_dir, metadata_rows)
