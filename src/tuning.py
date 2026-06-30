"""
tuning.py — Optuna wrapper for UniBFS parameter tuning (Phase 4).

Implements nested cross-validation:
  - Outer: 80/20 train/test split (mirrors paper's RQ4 setup).
  - Inner: Optuna tunes NSch, CHr, Ct, Nt using 5-fold CV on train set only.
  - Final score: accuracy on held-out test set with best params.

Usage:
    from src.tuning import tune_unibfs
    result = tune_unibfs(data, labels, n_trials=50, n_inner_runs=3)
"""

import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from typing import Optional

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _inner_objective(trial, X_train, y_train, n_inner_runs: int = 3, use_relifish: bool = False) -> float:
    """Optuna objective: evaluate param set on training data via mini runs."""
    from .unibfs import run_unibfs

    params = {
        "NSch": trial.suggest_float("NSch", 0.1, 0.95),
        "CHr":  trial.suggest_float("CHr",  0.001, 0.3, log=True),
        "Ct":   trial.suggest_int("Ct", 100, 1000),
        "Nt":   trial.suggest_int("Nt", 10, 400),
    }

    result = run_unibfs(
        data=X_train,
        labels=y_train,
        MaxFEs=2000,           # reduced FEs for inner tuning speed
        Max_Run=n_inner_runs,  # fewer runs for speed
        verbose=False,
        n_jobs=-1,             # parallel runs within each trial
        use_relifish=use_relifish,
        **params,
    )
    return result["mean_acc"]


def tune_unibfs(
    data: np.ndarray,
    labels: np.ndarray,
    n_trials: int = 50,
    n_inner_runs: int = 5,  # Increased for variance smoothing
    test_size: float = 0.2,
    MaxFEs_final: int = 6000,
    Max_Run_final: int = 10,
    use_relifish: bool = False,
    seed: Optional[int] = None,
) -> dict:
    """Tune UniBFS with nested CV then evaluate on hold-out test set."""
    if seed is not None:
        np.random.seed(seed)

    # Outer split: 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size,
        random_state=seed, stratify=labels if _can_stratify(labels) else None,
    )

    # Inner tuning with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _inner_objective(trial, X_train, y_train, n_inner_runs, use_relifish),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_params = study.best_params

    # Final evaluation on test set using best params
    from .unibfs import run_unibfs

    # Train on X_train with best params, then evaluate on X_test
    result = run_unibfs(
        data=X_train,
        labels=y_train,
        MaxFEs=MaxFEs_final,
        Max_Run=Max_Run_final,
        verbose=False,
        n_jobs=-1,
        use_relifish=use_relifish,
        **best_params,
    )

    # Evaluate each run's best mask on X_test
    test_accs = []
    for X_mask in result["final_X"]:
        sel = np.where(X_mask == 1)[0]
        if len(sel) == 0:
            test_accs.append(0.0)
            continue
        from sklearn.svm import LinearSVC
        model = LinearSVC(dual=False)
        model.fit(X_train[:, sel], y_train)
        preds = model.predict(X_test[:, sel])
        test_accs.append(float(np.mean(preds == y_test) * 100))

    return {
        "best_params":      best_params,
        "test_acc_mean":    float(np.mean(test_accs)),
        "test_acc_std":     float(np.std(test_accs)),
        "test_accs_list":   test_accs,
        "test_nfeats_mean": result["mean_nfeats"],
        "study":            study,
        "train_result":     result,
    }


def _can_stratify(labels: np.ndarray) -> bool:
    """Check if stratified split is possible (each class >=2 samples)."""
    unique, counts = np.unique(labels, return_counts=True)
    return bool(np.all(counts >= 2))
