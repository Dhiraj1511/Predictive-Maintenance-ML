from pathlib import Path
import zipfile
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_PATH = RAW_DIR / "ai4i2020.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

def load_dataset():
    """Load the official UCI AI4I CSV; download it automatically if missing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        zip_path = RAW_DIR / "ai4i_dataset.zip"
        try:
            print("Downloading the official UCI AI4I dataset...")
            urllib.request.urlretrieve(DATA_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    raise FileNotFoundError("The downloaded archive contains no CSV file.")
                # The official archive contains the AI4I dataset CSV.
                chosen = next((n for n in csv_names if "predictive" in n.lower()), csv_names[0])
                with zf.open(chosen) as source:
                    df = pd.read_csv(source)
                df.to_csv(DATA_PATH, index=False)
            zip_path.unlink(missing_ok=True)
        except Exception as exc:
            raise RuntimeError(
                "Dataset download failed. Download the AI4I 2020 dataset from "
                "https://archive.ics.uci.edu/dataset/601/ai4i and place its CSV at "
                f"{DATA_PATH}. Original error: {exc}"
            ) from exc
    return pd.read_csv(DATA_PATH)

def clean_column_names(df):
    """Standardize column names without changing their meaning."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out
