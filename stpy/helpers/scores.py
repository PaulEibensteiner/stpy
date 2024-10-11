import torch


def r_score_std(y_true, y_pred, std, alpha=1.0):
    return 1 - torch.mean((y_true - y_pred) ** 2) / (alpha * std**2)
