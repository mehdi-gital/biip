# call model and put it through a loop, write artifacts and pickle
import os
import time
import torch
from biip.core.model import NeuralBIIP
from biip.utils.data import get_batch
from biip.utils.validation import predict, rmse_loss
from biip.utils.visualization import plot_true_pred_cylinder, plot_loss_curve
from biip.utils.helpers import get_device


class Learner:
    def __init__(
            self,
            input_dim,
            hidden_dim,
            use_adjoint,
            activation_fn,
            learning_rate,
            weight_decay,
            device,
            artifacts_path,
            logger
    ):
        self.artifacts_path = artifacts_path
        self.device = get_device(device)
        self.logger = logger

        # initialize model
        self.model = NeuralBIIP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_adjoint=use_adjoint,
            activation_fn=activation_fn,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss = torch.nn.MSELoss()

        # training state
        self.train_losses = []
        self.validation_losses = []
        self.best_validation_rmse = float('inf')

    def fit(self, dataset, epochs, batch_size, batch_time):
        # training loop
        for epoch in range(0, epochs):  # model_hyperparameters['epoch']):
            start_time = time.time()
            # get batch data
            batch_f0, batch_t, batch_f, batch_f_boundary, batches = get_batch(
                dataset=dataset,
                batch_size=batch_size,
                batch_time=batch_time,
            )
            # forward pass of the model
            self.model.train()
            f_hat = self.model(
                timestamps=batch_t.to(self.device),
                f0_interior=batch_f0.to(self.device),
                regular_edge_index=dataset['regular_edge_index'].to(self.device),
                f_boundary=batch_f_boundary.to(self.device),
                half_edge_index=dataset['half_edge_index'].to(self.device),
            )
            # loss
            batch_loss = self.loss(f_hat, batch_f.to(self.device))
            self.train_losses.append(batch_loss.item())

            # backpropagation
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # validation
            self.model.eval()
            f_hat_total = predict(self.model, dataset, self.device)
            validation_rmse = rmse_loss(f_hat_total.squeeze(), dataset['f_interior'].squeeze().to(self.device))
            self.validation_losses.append(validation_rmse.item())

            self.logger.info('[*] epoch: {}, train_loss: {:.4f}, validation_rmse: {:.4f}, time: {:.1f}'.format(
                epoch, batch_loss, validation_rmse, time.time() - start_time))

            # best model
            self._save_model(validation_rmse)

    def _save_model(self, validation_rmse):
        # saving model with the best validation loss
        if validation_rmse.item() < self.best_validation_rmse:
            torch.save(self.model.state_dict(), os.path.join(self.artifacts_path, 'model', 'model.pt'))
            self.best_validation_rmse = validation_rmse.item()
            self.logger.info('[*] model saved.')

    def _load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.artifacts_path, 'model', 'model.pt')))

    def predict_train(self, dataset, grid_size_i, grid_size_j):
        self._load_model()
        self.model.eval()
        f_hat_train = predict(self.model, dataset, self.device)
        plot_true_pred_cylinder(
            title='train',
            grid_size_i=grid_size_i,
            grid_size_j=grid_size_j,
            dataset=dataset,
            f_hat=f_hat_train,
            save_path=os.path.join(self.artifacts_path, 'train/train.png')
        )

    def plot_loss_curves(self):
        plot_loss_curve(
            losses=self.train_losses,
            title='Training loss',
            ylabel='Log MSE loss',
            save_path=os.path.join(self.artifacts_path, 'train/training_loss.png')
        )
        plot_loss_curve(
            losses=self.validation_losses,
            title='Validation loss',
            ylabel='Log RMSE loss',
            save_path=os.path.join(self.artifacts_path, 'train/validation_loss.png')
        )




