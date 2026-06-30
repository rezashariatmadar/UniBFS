import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.neighbors import NearestNeighbors

def fisher_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns indices of features sorted by Fisher Score (ANOVA F-value) descending."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F_vals, _ = f_classif(X, y)
    F_vals = np.nan_to_num(F_vals)
    return np.argsort(F_vals)[::-1]

def relieff_score(X: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
    """Returns indices of features sorted by ReliefF weights descending."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    classes = np.unique(y)
    
    # Very basic ReliefF logic for 2+ classes
    nn = NearestNeighbors(n_neighbors=k+1, metric='manhattan')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    for i in range(n_samples):
        c = y[i]
        # Hits (same class)
        hits = [idx for idx in indices[i][1:] if y[idx] == c][:k]
        if hits:
            weights -= np.sum(np.abs(X[i] - X[hits]), axis=0) / (n_samples * k)
            
        # Misses (different classes)
        for diff_c in classes:
            if diff_c == c: continue
            misses = [idx for idx in indices[i][1:] if y[idx] == diff_c][:k]
            if misses:
                p_c = np.sum(y == diff_c) / n_samples
                weights += (p_c * np.sum(np.abs(X[i] - X[misses]), axis=0)) / (n_samples * k)

    return np.argsort(weights)[::-1]
