import numpy as np
from src.heuristics import fisher_score, relieff_score

def test_fisher_score():
    X = np.array([[1, 2], [1, 2], [9, 8], [9, 8]])
    y = np.array([0, 0, 1, 1])
    ranks = fisher_score(X, y)
    assert len(ranks) == 2
    assert isinstance(ranks, np.ndarray)

def test_relieff_score():
    X = np.array([[1, 2], [1, 2], [9, 8], [9, 8]])
    y = np.array([0, 0, 1, 1])
    ranks = relieff_score(X, y, k=2)
    assert len(ranks) == 2
    assert isinstance(ranks, np.ndarray)
