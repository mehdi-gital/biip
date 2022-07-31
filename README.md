# biip: boundary informed inverse pDE solvers on discretized compact Riemann surfaces

Paper: [https://arxiv.org/abs/2206.02911](https://arxiv.org/abs/2206.02911)

A graph neural network is combined with a neural ordinary differential equation solver to learn an unknown dynamical system on a surface with boundary. 

The content of this repo consist of three main parts:
- Synthetic data generation based on graph Laplacians
- Training in Torch (torchdiffeq and torch_geometric) and logging
- Inference, evaluation, etc

## Instructions

- Place the following training data and configs under `./data/[project-name]`
    - hyperparameters and configs in `configs.json`
    - nodes and edges of the graph and its boundary in `graph_data.json`
    - a tensor with the values of the field on the interior of the graph
    - a tensor with the values of the observations on the boundary of the graph
- Make the necessary changes in `train.py` and run
- Point to the correct model in `inference.py` and run

## Data
To run _biip_, you need to prepare two types of data structures; graph structure data, and
scalar field data that is defined on the nodes of the graph and change over time.

**Graph structure data** 

`graph_data.json`: contains the information of the graph structure and has the following keys:
- `interior_nodes`: List of interior node indices in the range of `0` to `num_interior_nodes`.   
- `boundary_nodes`: List of boundary node indices in the range of `num_interior_nodes` to `num_nodes`.
- `regular_edges`: List of interior edges of the graph. Each edge is represented as a list containing source 
and target node indices that are both interior nodes.
- `half_edges`: List of the _half_ edges of the graph. Each half edge is represented as a list
of source and target node indices where the source node is a boundary node and the target node is an
interior node.

Please refer to _Definition 3_ of the paper for more details.

**Note**: `regualr_edges` must be directed from boundary to interior nodes. This is
because _biip_ directly uses the scalar field data of the graph boundary in the
forward pass of the model (teacher forcing technique), and boundary nodes
don't receive any messages from the interior nodes in the message passing neural network.

**Scalar field data**

Scalar field data consists of:
- `f_interior.npy`: scalar field of the interior nodes over time. Its shape
is `[num_timestamps, num_interior_nodes, input_dim]`.
- `f_boundary.npy` scalar field of the interior nodes over time. Its shape
is `[num_timestamps, num_boundary_nodes, input_dim]`.
- `t.npy` observation timestamps. Its shape is `[num_timestamps]`.



