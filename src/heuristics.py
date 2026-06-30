"""
heuristics.py — Filter-based feature evaluation methods for UniBFS-ReliFish.

Implements Algorithm 2 (Fisher Score) and Algorithm 3 (ReliefF)
to intelligently guide the feature selection process.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def fisher_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the Fisher Score for each feature (Algorithm 2).
    
    Fisher Score(f_i) = sum(n_j * (mu_ij - mu_i)^2) / sum(n_j * sigma_ij^2)
    """
    classes = np.unique(y)
    n_features = X.shape[1]
    
    mu_global = np.mean(X, axis=0)
    
    numerator = np.zeros(n_features)
    denominator = np.zeros(n_features)
    
    for c in classes:
        X_c = X[y == c]
        n_c = X_c.shape[0]
        
        mu_c = np.mean(X_c, axis=0)
        sigma_c_sq = np.var(X_c, axis=0)
        
        numerator += n_c * (mu_c - mu_global)**2
        denominator += n_c * sigma_c_sq
        
    # Avoid division by zero
    denominator[denominator == 0] = 1e-12
    
    scores = numerator / denominator
    return scores

def relieff_score(X: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Calculate the ReliefF Score for each feature (Algorithm 3).
    """
    n_samples, n_features = X.shape
    classes, counts = np.unique(y, return_counts=True)
    prob_class = {c: count / n_samples for c, count in zip(classes, counts)}
    
    weights = np.zeros(n_features)
    
    # Precompute nearest neighbors within the same class (Hits)
    hits = {}
    for c in classes:
        X_c = X[y == c]
        if X_c.shape[0] > k:
            nn = NearestNeighbors(n_neighbors=k+1, metric='manhattan')
            nn.fit(X_c)
            hits[c] = nn.kneighbors(X_c, return_distance=False)[:, 1:] # exclude self
        else:
            # Fallback if class has <= k samples
            n_neighbors = max(1, X_c.shape[0] - 1)
            nn = NearestNeighbors(n_neighbors=n_neighbors+1, metric='manhattan')
            nn.fit(X_c)
            hits[c] = nn.kneighbors(X_c, return_distance=False)[:, 1:]
            
    # Precompute nearest neighbors in other classes (Misses)
    misses = {c: {} for c in classes}
    for c in classes:
        for other_c in classes:
            if c == other_c:
                continue
            X_other = X[y == other_c]
            n_neighbors = min(k, X_other.shape[0])
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='manhattan')
            nn.fit(X_other)
            misses[c][other_c] = nn.kneighbors(X[y == c], return_distance=False)
            
    # Range of features for normalization
    feat_max = np.max(X, axis=0)
    feat_min = np.min(X, axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0 # avoid div by zero
    
    m = n_samples # execute m times
    
    # Iterate over all samples
    for i in range(m):
        c = y[i]
        
        # Get hits
        idx_in_class = np.sum(y[:i] == c) # get relative index for hits array
        if c in hits and hits[c].shape[0] > idx_in_class:
            hit_indices = hits[c][idx_in_class]
            hit_diffs = np.abs(X[i] - X[y == c][hit_indices]) / feat_range
            hit_penalty = np.mean(hit_diffs, axis=0)
            weights -= hit_penalty / m
            
        # Get misses
        for other_c in classes:
            if c == other_c:
                continue
                
            if other_c in misses[c] and misses[c][other_c].shape[0] > idx_in_class:
                miss_indices = misses[c][other_c][idx_in_class]
                miss_diffs = np.abs(X[i] - X[y == other_c][miss_indices]) / feat_range
                miss_reward = np.mean(miss_diffs, axis=0)
                
                # Weight by probability
                prob_factor = prob_class[other_c] / (1.0 - prob_class[c])
                weights += (prob_factor * miss_reward) / m
                
    return weights
