import glob
import os


OFFICIAL_VERTICAL_ORDER_CARDIAC_CSV_FILES = {"A4C_train.csv", "PSAX_train.csv"}


def collect_effective_train_csvs(data_root: str, csv_path: str, verbose: bool = True) -> list[str]:
    del data_root
    csv_files = sorted(glob.glob(os.path.join(csv_path, "*.csv")))
    if verbose:
        for csv_file in csv_files:
            if os.path.basename(csv_file) in OFFICIAL_VERTICAL_ORDER_CARDIAC_CSV_FILES:
                print(f"Using organizer-confirmed vertical-order CSV: {csv_file}")
    return csv_files
