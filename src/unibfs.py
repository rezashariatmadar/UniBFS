import numpy as np
import math
from .fitness import fitness
from .rfe import rfe

def _single_run(data: np.ndarray, labels: np.ndarray, MaxFEs: int, 
                NSch: float, CHr: float, Ct: int, Nt: int, 
                verbose: bool, top_fisher: np.ndarray = None, top_relieff: np.ndarray = None) -> dict:
    n_samples, Nvar = data.shape
    
    BestCost_col = np.zeros(MaxFEs + 1)
    SF_Best_Sol_col = np.zeros(MaxFEs + 1)
    
    EFs = 1
    counter = 0
    
    if top_fisher is not None and top_relieff is not None:
        X1 = np.random.randint(0, 2, Nvar)
        X2 = np.zeros(Nvar, dtype=int)
        X2[top_relieff] = 1
        X3 = np.zeros(Nvar, dtype=int)
        X3[top_fisher] = 1
        
        fit1 = fitness(X1, data, labels, k_neighbors=1)
        fit2 = fitness(X2, data, labels, k_neighbors=1)
        fit3 = fitness(X3, data, labels, k_neighbors=1)
        
        fits = [fit1, fit2, fit3]
        Xs = [X1, X2, X3]
        best_idx = np.argmax(fits)
        X = Xs[best_idx].copy()
        Fit_X = fits[best_idx]
    else:
        X = np.random.randint(0, 2, Nvar)
        if np.sum(X) == 0:
            X[np.random.randint(0, Nvar)] = 1
        Fit_X = fitness(X, data, labels, k_neighbors=1)
        
    BestCost_col[EFs] = Fit_X
    SF_Best_Sol_col[EFs] = int(np.sum(X))
    
    while EFs <= MaxFEs:
        X_New = X.copy()
        
        if np.random.rand() > NSch:
            S_Index = np.where(X == 0)[0]
            if len(S_Index) > 0:
                if top_fisher is not None and top_relieff is not None and np.random.rand() <= 0.5:
                    if np.random.rand() > 0.5:
                        valid_relief = [f for f in top_relieff if X[f] == 0]
                        if len(valid_relief) > 0:
                            K = np.random.choice(valid_relief)
                        else:
                            K = np.random.choice(S_Index)
                    else:
                        valid_fisher = [f for f in top_fisher if X[f] == 0]
                        if len(valid_fisher) > 0:
                            K = np.random.choice(valid_fisher)
                        else:
                            K = np.random.choice(S_Index)
                else:
                    K = np.random.choice(S_Index)
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
               Max_Run: int = 1, n_jobs: int = 1, verbose: bool = True, use_relifish: bool = False) -> dict:
    from joblib import Parallel, delayed
    
    top_fisher = None
    top_relieff = None
    if use_relifish:
        from .heuristics import fisher_score, relieff_score
        f_scores = fisher_score(data, labels)
        r_scores = relieff_score(data, labels)
        
        k_top = min(200, data.shape[1])
        top_fisher = np.argsort(f_scores)[-k_top:]
        top_relieff = np.argsort(r_scores)[-k_top:]
        
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_single_run)(data, labels, MaxFEs, NSch, CHr, Ct, Nt, verbose, top_fisher, top_relieff)
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
