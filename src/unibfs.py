import numpy as np
import math
from .fitness import fitness
from .rfe import rfe

def _single_run(data: np.ndarray, labels: np.ndarray, MaxFEs: int, 
                NSch: float, CHr: float, Ct: int, Nt: int, 
                ranks_relief: np.ndarray, ranks_fisher: np.ndarray,
                verbose: bool) -> dict:
    n_samples, Nvar = data.shape
    
    BestCost_col = np.zeros(MaxFEs + 1)
    SF_Best_Sol_col = np.zeros(MaxFEs + 1)
    
    EFs = 1
    counter = 0
    
    X = np.random.randint(0, 2, Nvar)
    if np.sum(X) == 0:
        X[np.random.randint(0, Nvar)] = 1
        
    Fit_X = fitness(X, data, labels, k_neighbors=1)
    BestCost_col[EFs] = Fit_X
    SF_Best_Sol_col[EFs] = int(np.sum(X))
    EFs += 1
    
    X11 = np.zeros(Nvar, dtype=int)
    Fit_X11 = 0.0
    X22 = np.zeros(Nvar, dtype=int)
    Fit_X22 = 0.0
    
    scan_limit = min(200, Nvar)
    for i in range(1, scan_limit + 1):
        if EFs > MaxFEs: break
        
        X1 = np.zeros(Nvar, dtype=int)
        X1[ranks_relief[:i]] = 1
        f1 = fitness(X1, data, labels, k_neighbors=1)
        if f1 > Fit_X11:
            X11 = X1.copy()
            Fit_X11 = f1
            
        X2 = np.zeros(Nvar, dtype=int)
        X2[ranks_fisher[:i]] = 1
        f2 = fitness(X2, data, labels, k_neighbors=1)
        if f2 > Fit_X22:
            X22 = X2.copy()
            Fit_X22 = f2
            
        best_overall_fit = max(Fit_X11, Fit_X22, Fit_X)
        best_overall_len = int(np.sum(X11)) if best_overall_fit == Fit_X11 else (int(np.sum(X22)) if best_overall_fit == Fit_X22 else int(np.sum(X)))
        
        BestCost_col[EFs] = best_overall_fit
        SF_Best_Sol_col[EFs] = best_overall_len
        EFs += 1
        
    best_idx = np.argmax([Fit_X11, Fit_X22, Fit_X])
    if best_idx == 0:
        X, Fit_X = X11, Fit_X11
    elif best_idx == 1:
        X, Fit_X = X22, Fit_X22
    
    while EFs <= MaxFEs:
        X_New = X.copy()
        
        if np.random.rand() > NSch:
            # Guided Addition (ReliFish)
            k_rand = np.random.randint(0, min(100, Nvar))
            if np.random.rand() > 0.5:
                K = ranks_relief[k_rand]
            else:
                K = ranks_fisher[k_rand]
            X_New[K] = 1
        else:
            nmu = int(np.ceil(np.random.rand() * Nvar))
            j = np.random.choice(Nvar, min(nmu, Nvar), replace=False)
            X_New[j] = 0
            
        if np.sum(X_New) == 0:
            X_New = X.copy()
            X_New[np.random.randint(0, Nvar)] = 1
            
        Fit_X_New = fitness(X_New, data, labels, k_neighbors=1)
        
        if Fit_X_New > Fit_X:
            counter = 0
        else:
            counter += 1
                
        if Fit_X_New >= Fit_X:
            X = X_New
            Fit_X = Fit_X_New
            
        if EFs <= MaxFEs:
            BestCost_col[EFs] = Fit_X
            SF_Best_Sol_col[EFs] = int(np.sum(X))
            
        if verbose:
            print(f"UniBFS: EF={EFs}  Acc={Fit_X:.2f}  #Feat={int(np.sum(X))}")
            
        if counter >= Ct and np.sum(X) >= Nt:
            sel_idx = np.where(X == 1)[0]
            data_sub = data[:, sel_idx]
            
            best_idx, best_fit, EFs, rfe_hist = rfe(
                data_sub=data_sub,
                labels=labels,
                original_indices=sel_idx,
                Fit_X=Fit_X,
                EFs=EFs,
                Max_FEs_LSA=500,
                CHr=CHr,
                verbose=verbose
            )
            
            X = np.zeros(Nvar, dtype=int)
            X[best_idx] = 1
            Fit_X = fitness(X, data, labels, k_neighbors=1)
            counter = 0
            
            for ef_step, fit_val, n_feat in rfe_hist:
                if ef_step <= MaxFEs:
                    BestCost_col[ef_step] = fit_val
                    SF_Best_Sol_col[ef_step] = n_feat
                    
        EFs += 1
        
    return {
        "final_X": X,
        "BestCost": BestCost_col,
        "SF_Best_Sol": SF_Best_Sol_col,
        "final_acc": Fit_X,
        "final_nfeats": int(np.sum(X))
    }

def run_unibfs(data: np.ndarray, labels: np.ndarray, MaxFEs: int = 6000, 
               NSch: float = 0.8, CHr: float = 0.01, Ct: int = 500, Nt: int = 50, 
               Max_Run: int = 1, n_jobs: int = 1, verbose: bool = True) -> dict:
    from joblib import Parallel, delayed
    from .heuristics import fisher_score, relieff_score
    
    ranks_relief = relieff_score(data, labels, k=10)
    ranks_fisher = fisher_score(data, labels)
    
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_single_run)(data, labels, MaxFEs, NSch, CHr, Ct, Nt, ranks_relief, ranks_fisher, verbose)
        for _ in range(Max_Run)
    )
    
    final_Xs = [r["final_X"] for r in results]
    accs = [r["final_acc"] for r in results]
    nfeats = [r["final_nfeats"] for r in results]
    
    return {
        "final_X": final_Xs,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_nfeats": float(np.mean(nfeats)),
        "runs": results
    }
