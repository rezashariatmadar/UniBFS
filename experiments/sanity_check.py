"""
sanity_check.py — Phase 2 sanity check.

Reproduces published UniBFS baseline numbers from Table 5 on small datasets.
Target: mean accuracy within ~1-2% of published values with DEFAULT params.

Published Table 5 targets (from paper):
  Colon:    Mean≈99.4, Std≈?, #Feat≈?
  GLIOMA:   Mean≈85.0  (approximate)
  Leukemia: Mean≈97.0  (approximate)

Usage:
    uv run python experiments/sanity_check.py
    uv run python experiments/sanity_check.py --dataset colon --runs 5

Data: place .mat files in the data/ directory.
  Colon.mat   -> from UniBFS-Algorithm repo (already present)
  GLIOMA.mat  -> download from scikit-feature datasets
"""

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.stats as stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.unibfs import run_unibfs


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_mat_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a .mat dataset file.

    Tries common key patterns found in UniBFS datasets:
      - 'data' column format: data[:,1:] = features, data[:,0] = labels
      - 'X' / 'Y' format (e.g. GLIOMA, leukemia_3)
    """
    from scipy.io import loadmat
    mat = loadmat(str(path))

    # Key pattern: X/Y
    if "X" in mat and "Y" in mat:
        X = np.asarray(mat["X"], dtype=float)
        y = np.asarray(mat["Y"], dtype=int).ravel()
        return X, y

    # Key pattern: 'data' column (label first)
    if "data" in mat:
        d = np.asarray(mat["data"], dtype=float)
        y = d[:, 0].astype(int)
        X = d[:, 1:]
        return X, y

    # Key pattern: 'M' format (Breast, Bladder)
    if "M" in mat:
        d = np.asarray(mat["M"], dtype=float)
        y = d[:, -1].astype(int)
        X = d[:, 1:-1]
        return X, y

    raise ValueError(
        f"Unknown .mat format in {path}. Keys: {[k for k in mat if not k.startswith('_')]}"
    )


def load_colon_from_repo() -> tuple[np.ndarray, np.ndarray]:
    """Load Colon.mat from the cloned UniBFS-Algorithm repo."""
    mat_path = PROJECT_ROOT / "UniBFS-Algorithm" / "Colon.mat"
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Colon.mat not found at {mat_path}. "
            "It should be in the cloned UniBFS-Algorithm repo."
        )
    return load_mat_dataset(mat_path)


DATASET_REGISTRY = {
    "colon":     lambda: load_colon_from_repo(),
    "cns":       lambda: load_mat_dataset(PROJECT_ROOT / "data" / "CNS.mat"),
    "leukemia3": lambda: load_mat_dataset(PROJECT_ROOT / "data" / "Leukemia_3.mat"),
    "glioma":    lambda: load_mat_dataset(PROJECT_ROOT / "data" / "GLIOMA.mat"),
    "leukemia1": lambda: load_mat_dataset(PROJECT_ROOT / "data" / "leukemia.mat"),
}

# Published paper targets (Table 5, approximate)
PAPER_TARGETS = {
    "colon":     {"mean_acc": 99.4},
    "glioma":    {"mean_acc": 85.0},
    "leukemia3": {"mean_acc": 97.0},
    "leukemia1": {"mean_acc": 94.3},
    "cns":       {"mean_acc": 83.0},
}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_sanity_check(dataset_name: str, n_runs: int, verbose: bool) -> dict:
    """Run sanity check for one dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}  |  Runs: {n_runs}")
    print(f"Default params: NSch=0.8, CHr=0.01, MaxFEs=6000, Ct=500, Nt=50")
    print("="*60)

    loader = DATASET_REGISTRY[dataset_name]
    X_raw, y = loader()
    # z-score normalize (matches MATLAB zscore(Input))
    X = stats.zscore(X_raw, axis=0)
    n_samples, n_features = X.shape
    print(f"Loaded: {n_samples} samples, {n_features} features, {len(np.unique(y))} classes")

    result = run_unibfs(
        data=X,
        labels=y,
        MaxFEs=6000,
        Max_Run=n_runs,
        NSch=0.8,
        CHr=0.01,
        Ct=500,
        Nt=50,
        verbose=verbose,
    )

    print(f"\n" + "-"*40)
    print(f"Results ({dataset_name.upper()}, {n_runs} runs):")
    print(f"  Mean Acc : {result['mean_acc']:.2f}")
    print(f"  Std Acc  : {result['std_acc']:.2f}")
    print(f"  Best Acc : {result['best_acc']:.2f}")
    print(f"  Worst Acc: {result['worst_acc']:.2f}")
    print(f"  Mean #Feat: {result['mean_nfeats']:.1f}")

    target = PAPER_TARGETS.get(dataset_name)
    if target:
        diff = result["mean_acc"] - target["mean_acc"]
        status = "✓ PASS" if abs(diff) <= 2.0 else "✗ FAIL (>2% gap)"
        print(f"\n  Paper target: {target['mean_acc']:.1f}")
        print(f"  Difference  : {diff:+.2f}%  →  {status}")

    return result


def main():
    parser = argparse.ArgumentParser(description="UniBFS Phase 2 sanity check")
    parser.add_argument(
        "--dataset", default="colon",
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to test (default: colon)"
    )
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs (paper uses 30; use 5 for quick check)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-FE verbose output")
    parser.add_argument("--save", action="store_true",
                        help="Save results JSON to results/")
    args = parser.parse_args()

    result = run_sanity_check(
        dataset_name=args.dataset,
        n_runs=args.runs,
        verbose=not args.quiet,
    )

    if args.save:
        out_path = PROJECT_ROOT / "results" / f"sanity_{args.dataset}_{args.runs}runs.json"
        out_path.parent.mkdir(exist_ok=True)
        # Can't JSON serialize numpy arrays, convert
        save_data = {k: v for k, v in result.items() if k not in ("BestCost", "SF_Best_Sol", "final_X")}
        save_data["BestCost_last_row"] = result["BestCost"][-1, :].tolist()
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
