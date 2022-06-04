import torch


def predict(model, dataset, device):
    with torch.no_grad():
        f_hat_total = model(
            timestamps=dataset['timestamps'].float().to(device),
            f0_interior=dataset['f0_interior'].unsqueeze(0).float().to(device),
            regular_edge_index=dataset['regular_edge_index'].to(device),
            f_boundary=dataset['f_boundary'].unsqueeze(1).float().to(device),
            half_edge_index=dataset['half_edge_index'].to(device)
        )
    return f_hat_total


def rmse_loss(pred_f, true_f):
    root_mse = torch.sqrt(((pred_f - true_f)**2).mean())
    return root_mse
