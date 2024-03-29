import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
from torch_geometric.nn import SAGEConv


class DiffOp(nn.Module):
    """
    Neural differential operator $\mathrm{D}_{\Theta}$ defined in Equation 13 with teacher forcing.
    This differential operator is Dirichlet boundary informed.
    """

    def __init__(self, input_dim, hidden_dim, activation_fn):
        super(DiffOp, self).__init__()

        self.mp1 = SAGEConv(in_channels=input_dim, out_channels=hidden_dim)
        self.mp2 = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.mp3 = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.mp4 = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.mp5 = SAGEConv(in_channels=hidden_dim, out_channels=input_dim)
        if activation_fn == "softplus":
            self.act = nn.Softplus()
        else:
            raise NotImplementedError

    def forward(self, t, state):
        # unpacking the state
        f0_interior, regular_edge_index, f_boundary, half_edge_index, timestamps = state
        regular_edge_index = regular_edge_index.long()
        half_edge_index = half_edge_index.long()
        num_nodes_interior = f0_interior.size(1)

        # linear interpolation on the boundary nodes and between consecutive timestamps
        f_boundary_t = self.interpolate_f_boundary(t, timestamps, f_boundary)

        # creating edge_index of $G=intG \cup \partial G$
        edge_index = torch.cat([regular_edge_index, half_edge_index], dim=1)

        # creating the initial scalar field of $G=intG \cup \partial G$
        x0 = torch.cat([f0_interior, f_boundary_t], dim=1)

        # message from the boundary nodes
        transformed_f_boundary_t = self.mp1.lin_l(f_boundary_t)

        # message passing layer 1 defined in Equation 17
        x1 = self.mp1(x0, edge_index)
        x1 = self.act(x1)
        x1 = torch.cat(
            [x1[:, :num_nodes_interior, :], transformed_f_boundary_t], dim=1
        )  # teacher forcing

        # message passing layer 2 defined in Equation 17
        x2 = self.mp2(x1, edge_index)
        x2 = self.act(x2)
        x2 = torch.cat(
            [x2[:, :num_nodes_interior, :], transformed_f_boundary_t], dim=1
        )  # teacher forcing

        # message passing layer 3 defined in Equation 17
        x3 = self.mp3(x2, edge_index)
        x3 = self.act(x3)
        x3 = torch.cat(
            [x3[:, :num_nodes_interior, :], transformed_f_boundary_t], dim=1
        )  # teacher forcing

        # message passing layer 4 defined in Equation 17
        x4 = self.mp4(x3, edge_index)
        x4 = self.act(x4)
        x4 = torch.cat(
            [x4[:, :num_nodes_interior, :], transformed_f_boundary_t], dim=1
        )  # teacher forcing

        # message passing layer 5 with identity activation
        x5 = self.mp5(x4, edge_index)

        # creating the time derivative state of intG
        dstate_dt = (
            x5[:, :num_nodes_interior, :],
            torch.zeros_like(state[1]),
            torch.zeros_like(state[2]),
            torch.zeros_like(state[3]),
            torch.zeros_like(state[4]),
        )

        return dstate_dt

    @staticmethod
    def interpolate_f_boundary(t, timestamps, f_boundary):
        """
        Linearly approximating boundary_value_t based on t, delta_t, boundary_values.
        """
        delta_t = timestamps[1] - timestamps[0]
        k = torch.div(t, delta_t, rounding_mode="floor").long()
        if k == timestamps.size(0) - 1:
            return f_boundary[k]
        min_time = timestamps[k]
        max_time = timestamps[k + 1]
        # scaling t to 0<t<1
        scaled_t = (t - min_time) / (max_time - min_time)
        # linear interpolation of boundary_values_t
        f_boundary_t = (1 - scaled_t) * f_boundary[k] + scaled_t * f_boundary[k + 1]
        return f_boundary_t

    def __repr__(self):
        return self.__class__.__name__


class IntOp(nn.Module):
    """
    Integration operator.
    """

    def __init__(self, diffop, use_adjoint):
        super(IntOp, self).__init__()
        self.diffop = diffop
        if use_adjoint:
            self.odeint = odeint_adjoint
        else:
            self.odeint = odeint

    def forward(
        self, timestamps, f0_interior, regular_edge_index, f_boundary, half_edge_index
    ):
        # packing the state
        initial_state = (
            f0_interior,
            regular_edge_index,
            f_boundary,
            half_edge_index,
            timestamps,
        )

        # integration
        updated_state = self.odeint(
            func=self.diffop,
            y0=initial_state,
            t=timestamps,
            method="dopri5",
            options={"step_t": timestamps[-1], "first_step": torch.tensor(0.0001)},
        )

        f_t = updated_state[0]
        return f_t

    def __repr__(self):
        return self.__class__.__name__


class NeuralBIIP(nn.Module):
    """
    Neural dynamical system defined in Equation 16.
    """

    def __init__(self, input_dim, hidden_dim, use_adjoint, activation_fn):
        super(NeuralBIIP, self).__init__()

        self.diffop = DiffOp(
            input_dim=input_dim, hidden_dim=hidden_dim, activation_fn=activation_fn
        )
        self.intop = IntOp(diffop=self.diffop, use_adjoint=use_adjoint)

    def forward(
        self, timestamps, f0_interior, regular_edge_index, f_boundary, half_edge_index
    ):
        f_t = self.intop(
            timestamps, f0_interior, regular_edge_index, f_boundary, half_edge_index
        )
        return f_t

    def predict(self, dataset, device):
        with torch.no_grad():
            f_t_hat = self.intop(
                timestamps=dataset["timestamps"].float().to(device),
                f0_interior=dataset["f0_interior"].unsqueeze(0).float().to(device),
                regular_edge_index=dataset["regular_edge_index"].to(device),
                f_boundary=dataset["f_boundary"].unsqueeze(1).float().to(device),
                half_edge_index=dataset["half_edge_index"].to(device),
            )
        return f_t_hat

    def get_vector_field(self, t, timestamps, f0_interior, regular_edge_index, f_boundary, half_edge_index):
        initial_state = (f0_interior, regular_edge_index, f_boundary, half_edge_index, timestamps)
        df_dt = self.diffop(t, initial_state)[0]
        return df_dt

    def __repr__(self):
        return self.__class__.__name__
