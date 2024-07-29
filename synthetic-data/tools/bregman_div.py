import numpy as np


def bregman_div(w1, w2, p):
    pos = (np.linalg.norm(w1.flatten(), p) ** p) / p + (p - 1) * (
        np.linalg.norm(w2.flatten(), p) ** p
    ) / p
    neg = np.sum(np.sign(w2) * (np.abs(w2) ** (p - 1)) * w1)
    return pos - neg


def bregman_correlation(w1, w2, p):
    return bregman_div(
        w1 / np.linalg.norm(w1.flatten(), p), w2 / np.linalg.norm(w2.flatten(), p), p
    )
