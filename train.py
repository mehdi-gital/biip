# reads data, loads models, loads configs, trains and creates model artifacts
import os
import json
from biip.core.learner import Learner
from biip.utils.data import prepare_dataset


def main():
    # paths
    project_name = 'experiment_0'
    data_path = os.path.join('data', project_name)
    artifacts_path = os.path.join('artifacts', project_name)

    # reading the configurations
    with open(os.path.join(data_path, 'configs.json')) as file:
        configs = json.load(file)

    # constructing the dataset
    train_dataset = prepare_dataset(data_path=data_path, mode='train')

    # initialize the learner
    learner = Learner(
        input_dim=configs['neuralBIIP']['input_dim'],
        hidden_dim=configs['neuralBIIP']['hidden_dim'],
        use_adjoint=configs['neuralBIIP']['use_adjoint'],
        activation_fn=configs['neuralBIIP']['activation_fn'],
        learning_rate=configs['training']['learning_rate'],
        weight_decay=configs['training']['weight_decay'],
        device=configs['training']['device'],
        artifacts_path=artifacts_path
    )
    # fit learner
    learner.fit(
        dataset=train_dataset,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        batch_time=configs['training']['batch_time']
    )


if __name__ == '__main__':
    main()
