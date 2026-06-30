import urllib.request
import os
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/"

DATASETS = [
    "GLIOMA.mat",
    "leukemia.mat",
    "11_Tumors.mat",
]

for ds in DATASETS:
    url = BASE_URL + ds
    out_path = DATA_DIR / ds
    if not out_path.exists():
        print(f"Downloading {ds} from {url} ...")
        try:
            urllib.request.urlretrieve(url, out_path)
            print(f"Saved to {out_path}")
        except Exception as e:
            print(f"Failed to download {ds}: {e}")
    else:
        print(f"{ds} already exists in {out_path}")
