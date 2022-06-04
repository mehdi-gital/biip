import torch


def rmse_loss(pred_f, true_f):
    root_mse = torch.sqrt(((pred_f - true_f)**2).mean())
    return root_mse
