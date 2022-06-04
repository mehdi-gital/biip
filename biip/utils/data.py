# reads graph data from json along with scalar field data and timestamps and initializes torch objects
import os
import json
import torch
import numpy as np


def prepare_dataset(data_path, prefix):
    # read graph data
    graph_data_path = os.path.join(data_path, 'graph_data.json')
    with open(graph_data_path) as file:
        graph_data = json.load(file)
    # read scalar field and timestamps data
    f_interior = np.load(os.path.join(data_path, f'{prefix}_f_interior.npy'))
    f_boundary = np.load(os.path.join(data_path, f'{prefix}_f_boundary.npy'))
    timestamps = np.load(os.path.join(data_path, f'{prefix}_t.npy'))

    # constructing the dataset
    regular_edge_index = torch.tensor(graph_data['regular_edges'], dtype=torch.long).t()  # interior edge_index
    half_edge_index = torch.tensor(graph_data['half_edges'], dtype=torch.long).t()  # boundary edge_index
    f_interior = torch.tensor(f_interior)
    f_boundary = torch.tensor(f_boundary)
    timestamps = torch.tensor(timestamps)
    dataset = {
        'num_nodes_interior': len(graph_data['interior_nodes']),
        'num_nodes_boundary': len(graph_data['boundary_nodes']),
        'num_timesteps': timestamps.size(0),
        'num_features': f_interior.size(-1),
        'regular_edge_index': regular_edge_index,
        'half_edge_index': half_edge_index,
        'f_interior': f_interior,
        'f0_interior': f_interior[0],
        'f_boundary': f_boundary,
        'timestamps': timestamps
    }
    return dataset


def get_batch(
        dataset: dict,
        batch_size: int,
        batch_time: int,
):
    """returns a random subsegment of the timed data and observes the initial value within the segment."""

    num_timesteps = dataset['num_timesteps']
    f_interior = dataset['f_interior']
    t = dataset['timestamps']
    f_boundary = dataset['f_boundary']

    batches = torch.from_numpy(
        np.random.choice(
            np.arange(num_timesteps - batch_time, dtype=np.int64), batch_size, replace=False)
    )  # (batch_size, )

    batch_t = t[:batch_time]  # (batch_time, )
    batch_f0 = f_interior[batches, :, :].float()  # (batch_size, N, D)

    batch_f = []
    batch_f_boundary = []
    for batch in batches:
        fi = []
        fb = []
        for t in range(batch_time):
            fi.append(f_interior[batch + t].unsqueeze(0))
            fb.append(f_boundary[batch + t].unsqueeze(0))
        fi = torch.cat(fi, dim=0).unsqueeze(1)
        fb = torch.cat(fb, dim=0).unsqueeze(1)
        batch_f.append(fi)
        batch_f_boundary.append(fb)
    batch_f = torch.cat(batch_f, dim=1)  # (batch_time, batch_size, N, D)
    batch_f_boundary = torch.cat(batch_f_boundary, dim=1)

    return batch_f0.float(), batch_t.float(), batch_f.float(), batch_f_boundary.float(), batches
