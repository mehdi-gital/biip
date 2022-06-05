import os
import json
import torch
from biip.core.model import NeuralBIIP
from biip.utils.data import prepare_dataset
from biip.utils.validation import rmse_loss
from biip.utils.visualization import plot_true_pred_cylinder
from biip.utils.helpers import get_device, get_logger


# paths
project_name = 'experiment_0'
data_path = os.path.join('data', project_name)
artifacts_path = os.path.join('artifacts', project_name)


def main():
    # getting logger
    logger = get_logger(log_path=artifacts_path, prefix='inference/logs')

    # reading the configurations
    with open(os.path.join(data_path, 'configs.json')) as file:
        configs = json.load(file)

    # constructing the datasets
    dataset_test = prepare_dataset(data_path=data_path, prefix='test')
    dataset_new = prepare_dataset(data_path=data_path, prefix='new')

    # device
    device = get_device(configs['training']['device'])

    # load the trained model
    model = NeuralBIIP(
        input_dim=configs['neuralBIIP']['input_dim'],
        hidden_dim=configs['neuralBIIP']['hidden_dim'],
        use_adjoint=configs['neuralBIIP']['use_adjoint'],
        activation_fn=configs['neuralBIIP']['activation_fn'],
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(artifacts_path, 'model/model.pt')))
    model.eval()

    # test data
    f_hat_test = model.predict(dataset_test, device)
    plot_true_pred_cylinder(
        title='test',
        grid_size_i=configs['dataset']['grid_size_i'],
        grid_size_j=configs['dataset']['grid_size_j'],
        dataset=dataset_test,
        f_hat=f_hat_test,
        save_path=os.path.join(artifacts_path, 'inference/test.png'),
    )
    rmse_test = rmse_loss(f_hat_test.squeeze(), dataset_test['f_interior'].squeeze().to(device))
    logger.info(f'Model RMSE on test dataset: {rmse_test.item()}')

    # new data
    f_hat_new = model.predict(dataset_new, device)
    plot_true_pred_cylinder(
        title='new',
        grid_size_i=configs['dataset']['grid_size_i'],
        grid_size_j=configs['dataset']['grid_size_j'],
        dataset=dataset_new,
        f_hat=f_hat_new,
        save_path=os.path.join(artifacts_path, 'inference/new.png'),
    )
    rmse_new = rmse_loss(f_hat_new.squeeze(), dataset_new['f_interior'].squeeze().to(device))
    logger.info(f'Model RMSE on new dataset: {rmse_new.item()}')


if __name__ == '__main__':
    main()
