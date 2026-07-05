import argparse
import os
import zipfile

from model import Model


def main():
    parser = argparse.ArgumentParser(description="Create challenge submission.zip from a trained checkpoint.")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="submission_output", help="Submission output directory")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the trained checkpoint to use for inference.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--split-csv", type=str, default=None, help="Optional CSV to restrict the prediction set")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = Model(checkpoint_path=args.checkpoint_path)
    json_path = model.predict(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        split_csv=args.split_csv,
        output_filename="regression_predictions.json",
    )

    zip_path = os.path.join(args.output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="regression_predictions.json")

    print(f"Submission JSON: {json_path}")
    print(f"Submission ZIP: {zip_path}")


if __name__ == "__main__":
    main()
