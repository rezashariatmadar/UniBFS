import sys
import os
import time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.unibfs import run_unibfs
from src.tuning import tune_unibfs
from src.stats import wilcoxon_test
from sklearn.model_selection import train_test_split

from scipy.io import loadmat

def load_data(filepath):
    mat = loadmat(filepath)
    df = mat['data']
    data = df[:, 1:].astype(float)
    data = stats.zscore(data)
    labels = df[:, 0]
    return data, labels

def evaluate_fixed(data, labels, NSch, Nt, Max_Run, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=seed, stratify=labels)
    
    result = run_unibfs(
        data=X_train,
        labels=y_train,
        MaxFEs=2000,
        Max_Run=Max_Run,
        NSch=NSch,
        Nt=Nt,
        verbose=False,
        n_jobs=-1
    )
    
    test_accs = []
    for X_mask in result["final_X"]:
        sel = np.where(X_mask == 1)[0]
        if len(sel) == 0:
            test_accs.append(0.0)
            continue
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        model.fit(X_train[:, sel], y_train)
        preds = model.predict(X_test[:, sel])
        test_accs.append(float(np.mean(preds == y_test) * 100))
        
    return test_accs, result["mean_nfeats"]

def main():
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'MLL.mat')
    if not os.path.exists(filepath):
        print(f"Data file not found: {filepath}")
        return
        
    data, labels = load_data(filepath)
    print(f"Loaded dataset: {data.shape[0]} instances, {data.shape[1]} features.")
    
    Max_Run = 10
    n_trials = 30  # Increased to allow Optuna to find the manual fix parameters
    
    print("\n--- 1. Running Fixed Default (NSch=0.8, Nt=50) ---")
    start = time.time()
    def_accs, def_nfeats = evaluate_fixed(data, labels, NSch=0.8, Nt=50, Max_Run=Max_Run)
    def_time = time.time() - start
    print(f"Default -> Acc: {np.mean(def_accs):.2f}% ± {np.std(def_accs):.2f} | Feats: {def_nfeats:.1f} | Time: {def_time:.1f}s")
    
    print("\n--- 2. Running Fixed Manual (NSch=0.2, Nt=300) ---")
    start = time.time()
    man_accs, man_nfeats = evaluate_fixed(data, labels, NSch=0.2, Nt=300, Max_Run=Max_Run)
    man_time = time.time() - start
    print(f"Manual  -> Acc: {np.mean(man_accs):.2f}% ± {np.std(man_accs):.2f} | Feats: {man_nfeats:.1f} | Time: {man_time:.1f}s")
    
    print("\n--- 3. Running Optuna Tuned ---")
    start = time.time()
    tuned_result = tune_unibfs(
        data=data,
        labels=labels,
        n_trials=n_trials,
        n_inner_runs=3,
        MaxFEs_final=2000,
        Max_Run_final=Max_Run,
        seed=42
    )
    tuned_time = time.time() - start
    tuned_accs = tuned_result["test_accs_list"]
    tuned_nfeats = tuned_result["test_nfeats_mean"]
    print(f"Tuned   -> Acc: {np.mean(tuned_accs):.2f}% ± {np.std(tuned_accs):.2f} | Feats: {tuned_nfeats:.1f} | Time: {tuned_time:.1f}s")
    print(f"Best Params found by Optuna: {tuned_result['best_params']}")
    
    print("\n--- 4. Statistical Validation (Wilcoxon) ---")
    w_tuned_vs_def = wilcoxon_test(tuned_accs, def_accs, alternative="greater")
    w_man_vs_def = wilcoxon_test(man_accs, def_accs, alternative="greater")
    
    print(f"Tuned > Default  : p-value = {w_tuned_vs_def['p_value']:.4f} (Significant: {w_tuned_vs_def['significant']})")
    print(f"Manual > Default : p-value = {w_man_vs_def['p_value']:.4f} (Significant: {w_man_vs_def['significant']})")

if __name__ == "__main__":
    main()
