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
