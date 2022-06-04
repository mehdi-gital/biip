# call model and put it through a loop, write artifacts and pickle
import os
import time
import torch
from biip.core.model import NeuralBIIP
from biip.utils.data import get_batch
from biip.utils.validation import rmse_loss


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
            artifacts_path
    ):
        self.artifacts_path = artifacts_path
        if device == 'cpu':
            self.device = torch.device('cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise ValueError('[!] Device is set to cuda in configs.json but cuda is not available. '
                             'Please set device to cpu.')

        # initialize model
        self.model = NeuralBIIP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_adjoint=use_adjoint,
            activation_fn=activation_fn,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss = torch.nn.MSELoss()

    def fit(self, dataset, epochs, batch_size, batch_time):
        # training loop
        losses = []
        validation_losses = []
        best_validation_rmse = float('inf')
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
            losses.append(batch_loss.item())

            # backpropagation
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # validation
            with torch.no_grad():
                self.model.eval()
                f_hat_total = self.model(
                    timestamps=dataset['timestamps'].float().to(self.device),
                    f0_interior=dataset['f0_interior'].unsqueeze(0).float().to(self.device),
                    regular_edge_index=dataset['regular_edge_index'].to(self.device),
                    f_boundary=dataset['f_boundary'].unsqueeze(1).float().to(self.device),
                    half_edge_index=dataset['half_edge_index'].to(self.device)
                )
            validation_rmse = rmse_loss(f_hat_total.squeeze(), dataset['f_interior'].squeeze().to(self.device))
            validation_losses.append(validation_rmse.item())

            print('[*] epoch: {}, train_loss: {:.4f}, validation_rmse: {:.4f}, time: {:.1f}'.format(
                epoch, batch_loss, validation_rmse, time.time() - start_time))

            # best model
            if validation_rmse.item() < best_validation_rmse:
                torch.save(self.model.state_dict(), os.path.join(self.artifacts_path, 'model', 'model.pt'))
                best_validation_rmse = validation_rmse.item()
                print('[*] model saved.')

