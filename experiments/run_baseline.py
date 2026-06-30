"""
run_baseline.py — Phase 5 targeted experiment.

Replicates Research Question 4 from the UniBFS paper:
  - Three conditions per dataset, 30 independent runs each:
      1. fixed_default: NSch=0.8, Nt=50   (paper baseline, shows generalization gap)
      2. fixed_manual:  NSch=0.2, Nt=300  (paper's manual fix, UniBFS-ReliFish-c)
      3. optuna_tuned:  params tuned per dataset via nested CV (our contribution)

  - Datasets: MLL, Lung_Cancer, Leukemia_GSE9476 (Table 9/10 from paper)
    These are the three datasets where the paper saw the worst gap.

  - Split: 80/20 train/test (mirrors paper's RQ4 setup).
  - Metrics: test accuracy, #selected features, wall-clock time.

Usage:
    # Pilot (10 runs, 30 Optuna trials — fast):
    uv run python experiments/run_baseline.py --runs 10 --optuna-trials 30

    # Full (30 runs, 50 Optuna trials — matches paper rigor):
    uv run python experiments/run_baseline.py --runs 30 --optuna-trials 50
"""

import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.stats as stats_module
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.unibfs import run_unibfs
from src.stats import wilcoxon_test, friedman_test
from src.tuning import tune_unibfs


# ── Dataset registry for RQ4 datasets ────────────────────────────────────────

def load_mat(path: Path):
    from scipy.io import loadmat
    mat = loadmat(str(path))
    if "X" in mat and "Y" in mat:
        return np.asarray(mat["X"], dtype=float), np.asarray(mat["Y"], dtype=int).ravel()
    if "data" in mat:
        d = np.asarray(mat["data"], dtype=float)
        return d[:, 1:], d[:, 0].astype(int)
    if "M" in mat:
        d = np.asarray(mat["M"], dtype=float)
        return d[:, 1:-1], d[:, -1].astype(int)
    raise ValueError(f"Unknown .mat format: {[k for k in mat if not k.startswith('_')]}")


# Map dataset names to their .mat files in data/
# NOTE: The RQ4 datasets (MLL, Lung_Cancer, Leukemia_GSE9476) need to be
# downloaded separately. Run: python download_datasets.py first.
# We fall back to available datasets (Colon, GLIOMA, CNS) if RQ4 sets missing.
RQ4_DATASETS = {
    "MLL":               "MLL.mat",
    "Lung_Cancer":       "LUNG.mat",
    "Leukemia":          "ALL_AML.mat",
}

FALLBACK_DATASETS = {
    "Colon":  "Colon.mat",
    "GLIOMA": "GLIOMA.mat",
    "CNS":    "CNS.mat",
}


def resolve_datasets(prefer_rq4: bool = True) -> dict:
    """Return {name: path} for available datasets."""
    datasets = {}
    candidates = RQ4_DATASETS if prefer_rq4 else FALLBACK_DATASETS
    for name, fname in candidates.items():
        path = PROJECT_ROOT / "data" / fname
        if path.exists():
            datasets[name] = path
    if not datasets:
        print("RQ4 datasets not found, falling back to available datasets.")
        for name, fname in FALLBACK_DATASETS.items():
            path = PROJECT_ROOT / "data" / fname
            if path.exists():
                datasets[name] = path
    return datasets


# ── Evaluation helper ─────────────────────────────────────────────────────────

def eval_on_test(X_masks, X_train, y_train, X_test, y_test):
    """Evaluate a list of binary feature masks on the test set."""
    accs, nfeats = [], []
    for mask in X_masks:
        sel = np.where(mask == 1)[0]
        if len(sel) == 0:
            accs.append(0.0)
            nfeats.append(0)
            continue
        mdl = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        mdl.fit(X_train[:, sel], y_train)
        preds = mdl.predict(X_test[:, sel])
        accs.append(float(np.mean(preds == y_test) * 100))
        nfeats.append(int(np.sum(mask)))
    return accs, nfeats


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(dataset_name: str, data_path: Path, n_runs: int,
                   optuna_trials: int, verbose: bool) -> dict:
    """Run all three conditions on one dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  |  Runs: {n_runs}  |  Optuna trials: {optuna_trials}")
    print("="*60)

    X_raw, y = load_mat(data_path)
    X = stats_module.zscore(X_raw, axis=0)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    # Outer 80/20 split (mirrors paper RQ4)
    can_strat = all(c >= 2 for c in np.unique(y, return_counts=True)[1])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if can_strat else None,
    )

    results = {}

    # ── 1. Fixed default ──────────────────────────────────────────────────────
    print(f"\n[1/3] Fixed default (NSch=0.8, Nt=50) ...")
    t0 = time.time()
    r_default = run_unibfs(X_train, y_train, MaxFEs=6000, Max_Run=n_runs,
                            NSch=0.8, CHr=0.01, Ct=500, Nt=50, verbose=verbose)
    accs_d, feats_d = eval_on_test(r_default["final_X"], X_train, y_train, X_test, y_test)
    results["fixed_default"] = {
        "test_accs": accs_d,
        "nfeats": feats_d,
        "mean_test_acc": float(np.mean(accs_d)),
        "std_test_acc":  float(np.std(accs_d)),
        "mean_nfeats":   float(np.mean(feats_d)),
        "time_sec":      time.time() - t0,
        "params":        {"NSch": 0.8, "CHr": 0.01, "Ct": 500, "Nt": 50},
    }
    print(f"     Test Acc: {results['fixed_default']['mean_test_acc']:.2f} "
          f"± {results['fixed_default']['std_test_acc']:.2f}  "
          f"| #Feat: {results['fixed_default']['mean_nfeats']:.1f}")

    # ── 2. Fixed manual (paper's hand-tuned fix) ──────────────────────────────
    print(f"\n[2/3] Fixed manual (NSch=0.2, Nt=300) ...")
    t0 = time.time()
    r_manual = run_unibfs(X_train, y_train, MaxFEs=6000, Max_Run=n_runs,
                           NSch=0.2, CHr=0.01, Ct=500, Nt=300, verbose=verbose)
    accs_m, feats_m = eval_on_test(r_manual["final_X"], X_train, y_train, X_test, y_test)
    results["fixed_manual"] = {
        "test_accs": accs_m,
        "nfeats": feats_m,
        "mean_test_acc": float(np.mean(accs_m)),
        "std_test_acc":  float(np.std(accs_m)),
        "mean_nfeats":   float(np.mean(feats_m)),
        "time_sec":      time.time() - t0,
        "params":        {"NSch": 0.2, "CHr": 0.01, "Ct": 500, "Nt": 300},
    }
    print(f"     Test Acc: {results['fixed_manual']['mean_test_acc']:.2f} "
          f"± {results['fixed_manual']['std_test_acc']:.2f}  "
          f"| #Feat: {results['fixed_manual']['mean_nfeats']:.1f}")

    # ── 3. Optuna-tuned ───────────────────────────────────────────────────────
    print(f"\n[3/3] Optuna-tuned ({optuna_trials} trials) ...")
    t0 = time.time()
    r_optuna = tune_unibfs(
        data=X, labels=y,
        n_trials=optuna_trials,
        n_inner_runs=3,
        test_size=0.2,
        MaxFEs_final=6000,
        Max_Run_final=n_runs,
        seed=42,
    )
    accs_o = r_optuna["train_result"]["final_X"]  # re-eval on same test set
    accs_o_test, feats_o = eval_on_test(
        r_optuna["train_result"]["final_X"], X_train, y_train, X_test, y_test
    )
    results["optuna_tuned"] = {
        "test_accs": accs_o_test,
        "nfeats": feats_o,
        "mean_test_acc": float(np.mean(accs_o_test)),
        "std_test_acc":  float(np.std(accs_o_test)),
        "mean_nfeats":   float(np.mean(feats_o)),
        "best_params":   r_optuna["best_params"],
        "time_sec":      time.time() - t0,
    }
    print(f"     Test Acc: {results['optuna_tuned']['mean_test_acc']:.2f} "
          f"± {results['optuna_tuned']['std_test_acc']:.2f}  "
          f"| #Feat: {results['optuna_tuned']['mean_nfeats']:.1f}")
    print(f"     Best params: {r_optuna['best_params']}")

    # ── Statistical tests ─────────────────────────────────────────────────────
    try:
        wt_d_vs_m = wilcoxon_test(accs_d, accs_m)
        wt_d_vs_o = wilcoxon_test(accs_d, accs_o_test)
        wt_m_vs_o = wilcoxon_test(accs_m, accs_o_test)
    except Exception as e:
        wt_d_vs_m = wt_d_vs_o = wt_m_vs_o = {"error": str(e)}

    results["stats"] = {
        "wilcoxon_default_vs_manual": wt_d_vs_m,
        "wilcoxon_default_vs_optuna": wt_d_vs_o,
        "wilcoxon_manual_vs_optuna":  wt_m_vs_o,
    }

    print(f"\n  Stats (Wilcoxon, p<0.05):")
    print(f"    Default vs Manual: p={wt_d_vs_m.get('p_value', 'N/A'):.4f}  sig={wt_d_vs_m.get('significant', '?')}")
    print(f"    Default vs Optuna: p={wt_d_vs_o.get('p_value', 'N/A'):.4f}  sig={wt_d_vs_o.get('significant', '?')}")
    print(f"    Manual  vs Optuna: p={wt_m_vs_o.get('p_value', 'N/A'):.4f}  sig={wt_m_vs_o.get('significant', '?')}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 5 targeted experiment")
    parser.add_argument("--runs",          type=int, default=10,
                        help="Runs per condition (paper: 30; pilot: 10)")
    parser.add_argument("--optuna-trials", type=int, default=30,
                        help="Optuna trials (paper: 50; pilot: 30)")
    parser.add_argument("--datasets",      nargs="+", default=None,
                        help="Override dataset names to run")
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()

    available = resolve_datasets()
    if args.datasets:
        available = {k: v for k, v in available.items() if k in args.datasets}

    if not available:
        print("No datasets found in data/. Run download_datasets.py first.")
        sys.exit(1)

    print(f"Datasets to run: {list(available.keys())}")

    all_results = {}
    for ds_name, ds_path in available.items():
        try:
            res = run_experiment(
                dataset_name=ds_name,
                data_path=ds_path,
                n_runs=args.runs,
                optuna_trials=args.optuna_trials,
                verbose=args.verbose,
            )
            all_results[ds_name] = res
        except Exception as e:
            print(f"ERROR on {ds_name}: {e}")
            import traceback; traceback.print_exc()

    # ── Cross-dataset Friedman test ───────────────────────────────────────────
    if len(all_results) >= 2:
        try:
            means_d = [all_results[ds]["fixed_default"]["mean_test_acc"] for ds in all_results]
            means_m = [all_results[ds]["fixed_manual"]["mean_test_acc"]  for ds in all_results]
            means_o = [all_results[ds]["optuna_tuned"]["mean_test_acc"]  for ds in all_results]
            ft = friedman_test(means_d, means_m, means_o)
            print(f"\n  Friedman test (3 methods, {len(all_results)} datasets): "
                  f"p={ft['p_value']:.4f}  sig={ft['significant']}")
            for ds in all_results:
                all_results[ds]["friedman"] = ft
        except Exception as e:
            print(f"Friedman test failed: {e}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = (PROJECT_ROOT / "results" /
                f"phase5_{args.runs}runs_{args.optuna_trials}trials.json")
    out_path.parent.mkdir(exist_ok=True)

    # Serialize (remove non-JSON-serializable numpy arrays from nested dicts)
    def clean(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"\nAll results saved to {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Dataset':<20} {'Default':>12} {'Manual':>12} {'Optuna':>12}")
    print("-"*70)
    for ds, res in all_results.items():
        d = res["fixed_default"]["mean_test_acc"]
        m = res["fixed_manual"]["mean_test_acc"]
        o = res["optuna_tuned"]["mean_test_acc"]
        winner = "Optuna" if o >= m else "Manual"
        print(f"{ds:<20} {d:>11.2f}% {m:>11.2f}% {o:>11.2f}%  <- {winner} wins")
    print("="*70)


if __name__ == "__main__":
    main()
