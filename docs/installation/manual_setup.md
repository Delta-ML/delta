# Manual Setup

This project has been fully tested on `Python 3.6.8` and  `TensorFlow 2.0.0` under `Ubuntu 18.04.2 LTS`.
We recommend that users use `Docker` or a virtual environment such as `conda` to install the python requirements.

## Conda Package Install 

### Build conda envs

```shell
conda create -p <path>/<env_name> python=3.6
source activate <path>/<env_name>
```

### Install Tensorflow

```shell
conda install tensorflow-gpu=2.0.0
```

### Install dependences

Delta dependient on third party tools, so when run the program, need blow to install tools:

activate the environment and use below

```shell
cd tools && make
```

##  Pip Install

For case you want install `Tensorflow Gpu  2.0.0`, under machine which has `Gpu Driver 410.48`.
It has problem of runtime not compariable with driver version, when isntall using [conda](#conda-package-install).
Then we can install tensorflow from `Pip` as below:

### Build conda envs

Same to [conda install](#conda-package-install).

### Install CUDA toolkit and CUDANN

See [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) for `CUDA Toolkit and Compatible Driver Version`.
See [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html) for `cuDNN For CUDA and NVIDIA Hardware`.

For Nvidia Driver Version: 418.67, CUDA Version: 10.1:
```shell
conda install cudatoolkit==10.1.168-0
conda install cupti=10.1.168-0
conda install cudnn==7.6.0
```

or 
```shell
conda install cudatoolkit==10.1
conda install cupti==10.1
conda install cudnn==7.6.0
```

For user in China, we can set conda mirror as below:

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

Other references:
[conda-forge](https://github.com/conda-forge/conda-forge.github.io/issues/687)
[tuna](https://github.com/tuna/issues/issues/507)


### Install Tensorflow

```shell
pip install  -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==2.0.0
```

For tensorflow 2.0.0, [make sure numpy version is 1.16.4](https://github.com/tensorflow/tensorflow/issues/31249).

### Install dependences

Same to [conda install](#conda-package-install).



## DELTA install

### NLP User

Install DELTA without `speech` dependences:

```shell
cd tools && make basic check_install

```


### Speech User

By default we will install DELTA with `Kaldi` toolkit:

```shell
cd tools && make delta
```

If user has installed `Kaldi`, please DELTA as below:

```shell
cd tools && make delta KALDI=<kaldi-path>
```

it is simply link the `<kaldi-path>` to `tools/kaldi`.

### Advanced User

Please see `delta` target of `tools/Makefile`.


## DELTANN install

Install DELTANN as below:

```
cd tools && make deltann
```

For more details, please see `deltann` target of `tools/Makefile`

