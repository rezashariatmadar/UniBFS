import numpy as np
import math
from .fitness import fitness

def rfe(data_sub: np.ndarray, labels: np.ndarray, original_indices: np.ndarray, 
        Fit_X: float, EFs: int, Max_FEs_LSA: int = 500, CHr: float = 0.01, verbose: bool = True) -> tuple:
    """Redundant Feature Elimination local search.
    
    Args:
        data_sub: Feature matrix restricted to currently selected features.
        labels: Label vector.
        original_indices: Column indices in the full dataset that correspond to data_sub columns.
        Fit_X: Fitness of the solution that triggered RFE.
        EFs: Current global function-evaluation counter.
        Max_FEs_LSA: Max additional FEs for this RFE call (default 500).
        CHr: Change rate (default 0.01).
        verbose: Print progress if True.
        
    Returns:
        best_original_indices: Indices (in full dataset) of features kept.
        best_fit: Fitness of best solution found.
        EFs: Updated function-evaluation counter.
        bookkeeping_history: A list of (EFs, fitness, num_selected) for tracking.
    """
    n_sub = data_sub.shape[1]
    X = np.ones(n_sub, dtype=int)
    
    # K=5 for RFE fitness as per MATLAB Fit_2.m
    Fit_X = fitness(X, data_sub, labels, k_neighbors=5)
    
    Max_EFs_target = EFs + Max_FEs_LSA
    history = []

    while EFs <= Max_EFs_target:
        EFs += 1
        X_New = X.copy()
        
        U_Index = np.where(X == 1)[0]
        NUSF = len(U_Index)
        if NUSF == 0:
            continue
            
        UN = math.ceil(NUSF * CHr)
        
        # Determine number of features to actually unselect (randperm(UN, 1) in MATLAB)
        # We sample uniformly between 1 and UN (inclusive)
        if UN > 1:
            UN = np.random.randint(1, UN + 1)
        else:
            UN = 1
            
        # Select EXACTLY UN features without replacement from U_Index
        UN = min(UN, NUSF) # safety bounds
        K = np.random.choice(U_Index, size=UN, replace=False)
        X_New[K] = 0
        
        Fit_X_New = fitness(X_New, data_sub, labels, k_neighbors=5)
        
        # >= logic strictly required
        if Fit_X_New >= Fit_X:
            X = X_New
            Fit_X = Fit_X_New
            
        history.append((EFs, Fit_X, int(np.sum(X))))
        
        if verbose:
            print(f"RFE: EF={EFs}  Acc={Fit_X:.2f}  #Features={int(np.sum(X))}")
            
    best_sub_idx = np.where(X == 1)[0]
    best_original_indices = original_indices[best_sub_idx]
    
    return best_original_indices, Fit_X, EFs, history
