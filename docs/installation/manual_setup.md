# Manual Setup

This project has been fully tested on `Python 3.6.8` and  `TensorFlow 1.14` under `Ubuntu 18.04.2 LTS`.
We recommend that users use `Docker` or a virtual environment such as `conda` to install the python requirements.

## Conda Package Install 

### Build conda envs

```shell
conda create -n <path>/<env_name> python=3.6
source activate <path>/<env_name>
```

### Install Tensorflow

```shell
conda install tensorflow-gpu=1.14.0
```

### Install dependences

Delta dependient on third party tools, so when run the program, need blow to install tools:

activate the environment and use below

```shell
cd tools && make
```

##  Pip Install

For case you want install `Tensorflow Gpu  1.14.0`, under machine which has `Gpu Driver 410.48`.
It has problem of runtime not compariable with driver version, when isntall using [conda](#conda-package-install).
Then we can install tensorflow from `Pip` as below:

### Build conda envs

Same to [conda install](#conda-package-install).

### Install CUDA toolkit and CUDANN

```shell
conda install cudatoolkit==10.0.130
conda install cudnn==7.6.0
```

### Install Tensorflow

```shell
pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==1.14.0
```

### Install dependences

Same to [conda install](#conda-package-install).
