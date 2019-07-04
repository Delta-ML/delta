# DELTA - a DEep Language Technology plAtform 

DELTA is a platform for building and depolying neural networks in Tensorflow.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Released Models](#released-models)
- [Experiments and Benchmarks](#experiments-and-benchmarks)
- [License](#license)
- [References](#references)

## Installation

### Easy installation
To install a basic version of DELTA, simply use a one-line command:

```shell
source install.sh
```

This command will create a conda environment with required packages and install a basic DELTA (support model training for NLP tasks).

To verify the installtion, run (make sure you have the data files)
```shell
python delta/main.py --cmd train_and_eval --config delta/config/han-cls-keras/han-cls2.yml
```

For advanced installation, refer to the following instruction.

### Tensorflow Env
This project has been fully tested on `Python 3.6.8` and  `TensorFlow 1.14` under `Ubuntu 18.04.2 LTS`.
We recommend that users use `Docker` or a virtual environment such as `conda` to install the python requirements.

```shell
conda create -n <path>/<env_name> python=3.6
source activate <path>/<env_name>
conda install  tensorflow-gpu=1.14.0
```

### DELTA Env

Delta dependient on third party tools, so when run the program, need blow to install tools:

activate the environment and use below

```shell
cd tools && make 
```

to install all tools and packages needed. 

When to using DELTA, user must activate DELTA env by:

```shell
source env.sh
```

## Quick Start

DELTA organized data preparation and model trainning under `egs` directory, and all configuration using `yaml`.

* preparing trainnig data and model configuration.  
* training model.  
* evaluation or inference model. 
* export saved model. 

DELTA-NN organized model deployment under `dpl` directory. 

* configure `model.yaml` and convert model to other deployment model.  
* compile `delta-nn` and prepare runtime.
* testing, benchmarking or serving under docker.

## Train and Test

### Check Configuration file

### Generate Vocabulary

### Start Train and Eval 

We use only one command to start model train

```shell
python delta/main.py --cmd train_and_eval --config <your configuration file>.yml
```

and

```shell
python delta/main.py --cmd eval --config <your configuration file>.yml 
```

evaluation procedure.

## Evaluation on Dev

In order to boost efficiency of train, we use unique evaluation strategy. 
Evaluation on dev should be ran in another program. 
The strategy is that evaluation program loads latest checkpoint produced by train program in every time intervals and save top-k checkpoints.
You can start this evaluation program by command:

```shell
python dev_main.py --config <your configuration file>.yml
```

## Experiments and Benchmarks

## Released Models
For more details, please refer to [released models](docs/released_models.md).

## TODO
For more details, please refer to [TODO list](docs/todo.md).

## Contributing
Any kind of contribution is welcome. Please just open an issue without any hesitation.
For more details, please refer to [the contribution guide.](CONTRIBUTING).

## License
The DELTA platform is licensed under the terms of the Apache license.
See [LICENSE](LICENSE) for more information.

## Acknowlegement 
The DELTA platform depends on many open source repos.
See [References](docs/references.rst) for more information.

