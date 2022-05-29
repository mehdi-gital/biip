# biip: boundary informed inverse pDE solvers on discretized compact Riemann surfaces

## synthetic data generation

- graph data.
- sheaf evolution data on the graph.
- data structure and formats.
- diffusion on graphs as a class objec. 
- each data type (grad f as vector field, etc) generator is its own class


## model training 

- create a requirements.txt file.
- create a learning class as subclass of nn or torchdiffeq or whatever makes sense.
- let's work with diffop (?) and intop (?)
- in an object oriented way, what's the relationship between odefunc and odeblock?
- what are some of the options? what are the expected data formats and tensor sizes? What data type is the graph?
- let's use a data object from torchgeometric.
- using the learning object, get hyperparamms and methods, train, pickle the params and write the logs in json. 
- leave a template for applying hyperparam optimization. 
- train model by loading data and write model params as a pickle. 

## model inference (+ CICD)

- take a pickle file name, load the model
- make an inference one timestamp into the future
- leave a template for installing model explainability
- collect the output logs with IDs.
- compute model performance and log the summary.
- retrain and write the artifacts somewhere
- add the new data to the new addition to the data

## visualizations

- plot line integrals over grids in plt.
