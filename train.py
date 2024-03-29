import os
import json
from biip.core.learner import Learner
from biip.utils.data import prepare_dataset
from biip.utils.visualization import plot_true_pred_cylinder
from biip.utils.helpers import get_logger

project_name = "experiment_0"
# paths
data_path = os.path.join("data", project_name)
artifacts_path = os.path.join("artifacts", project_name)


def main():
    # getting logger
    logger = get_logger(log_path=artifacts_path, prefix="train/logs")

    # reading configurations
    with open(os.path.join(data_path, "configs.json")) as file:
        configs = json.load(file)

    # constructing the dataset
    train_dataset = prepare_dataset(data_path=data_path, prefix="train")

    # initialize the learner
    learner = Learner(
        input_dim=configs["neuralBIIP"]["input_dim"],
        hidden_dim=configs["neuralBIIP"]["hidden_dim"],
        use_adjoint=configs["neuralBIIP"]["use_adjoint"],
        activation_fn=configs["neuralBIIP"]["activation_fn"],
        learning_rate=configs["training"]["learning_rate"],
        weight_decay=configs["training"]["weight_decay"],
        device=configs["training"]["device"],
        artifacts_path=artifacts_path,
        logger=logger,
    )
    # fit learner
    learner.fit(
        dataset=train_dataset,
        epochs=configs["training"]["epochs"],
        batch_size=configs["training"]["batch_size"],
        batch_time=configs["training"]["batch_time"],
    )
    # plot learning curves
    learner.plot_loss_curves()

    # validate model on entire train data
    model = learner.get_model()
    model.eval()
    f_hat_train = model.predict(train_dataset, learner.device)
    plot_true_pred_cylinder(
        title="train",
        grid_size_i=configs["dataset"]["grid_size_i"],
        grid_size_j=configs["dataset"]["grid_size_j"],
        dataset=train_dataset,
        f_hat=f_hat_train,
        save_path=os.path.join(artifacts_path, "train/train.png"),
    )


if __name__ == "__main__":
    main()
