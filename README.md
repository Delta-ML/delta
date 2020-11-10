<div align="center">
  <img src="docs/_static/delta_logo.png">
</div>


[![Build Status](https://travis-ci.org/didi/delta.svg?branch=master)](https://travis-ci.org/didi/delta)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub top language](https://img.shields.io/github/languages/top/didi/delta)](https://img.shields.io/github/languages/top/didi/delta)
[![GitHub Issues](https://img.shields.io/github/issues/didi/delta.svg)](https://github.com/didi/delta/issues)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/didi/delta/blob/master/LICENSE)

# DELTA - A DEep learning Language Technology plAtform


## What is DELTA?

**DELTA** is a deep learning based end-to-end **natural language and speech processing** platform. 
DELTA aims to provide easy and fast experiences for using, deploying, and developing natural language processing and speech models for both academia and industry use cases. DELTA is mainly implemented using TensorFlow and Python 3.

For details of DELTA, please refer to this [paper](https://arxiv.org/abs/1908.01853).

## What can DELTA do?

DELTA has been used for developing several state-of-the-art algorithms for publications and delivering real production to serve millions of users. 
It helps you to train, develop, and deploy NLP and/or speech models, featuring:

 - Easy-to-use
   - One command to train NLP and speech models, including:
     - NLP: text classification, named entity recognition, question and answering, text summarization, etc
     - Speech: speech recognition, speaker verification, emotion recognition, etc
   - Use configuration files to easily tune parameters and network structures
 - Easy-to-deploy
   - What you see in training is what you get in serving: all data processing and features extraction are integrated into a model graph
   - Uniform I/O interfaces and no changes for new models
 - Easy-to-develop
   - Easily build state-of-the-art models using modularized components
   - All modules are reliable and fully-tested

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmarks](#benchmarks)
- [FAQ](#faq)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)
- [Acknowledgement](#acknowledgement)

## Installation

We provide several approach to install DELTA:

- If you are only interested in NLP tasks, you can [use `pip` to install](#install-from-pip
) DELTA.

- If you are interested in both NLP and speech tasks, you can install DELTA [from the source code](#install-from-source-code).

- If you are interested in model deployment, you may install DELTA [from the source code](#install-from-source-code) or [from `docker`](#install-from-docker).

### Install from pip

We provide the pip install support for `nlp` version of DELTA.

**Note**: Users can still install DELTA from the source for both `nlp` and `speech` tasks.

We recommend to create [conda](https://conda.io/) or
[virtualenv](https://virtualenv.pypa.io/en/latest/) and install DELTA
from pip in the virtual environment. For example
```bash
conda create -n delta-pip-py3.6 python=3.6
conda activate delta-pip-py3.6
```

Please install TensorFlow 2.x if you have not installed it in your system.
```bash
pip install tensorflow
```

Then, simply install DELTA use the following command:
```bash
pip install delta-nlp
```

After install DELTA, you can follow this example to train NLP models or develop new models.
[A Text Classification Usage Example for pip users](docs/tutorials/training/text_class_pip_example.md)

### Install from Source Code

To install from the source code, we use [conda](https://conda.io/) to
install required packages. Please
[install conda](https://conda.io/en/latest/miniconda.html) if you do not
have it in your system.

Also, we provide two options to install DELTA, `nlp` version or `full`
version. `nlp` version needs minimal requirements and only installs NLP
related packages:

```shell
# Run the installation script for NLP version, with CPU or GPU.
cd tools
./install/install-delta.sh nlp [cpu|gpu]
```

**Note**: Users from mainland China may need to set up conda mirror sources, see [./tools/install/install-delta.sh](tools/install/install-delta.sh) for details.

If you want to use both NLP and speech packages, you can install the `full` version. The full version needs [Kaldi](https://github.com/kaldi-asr/kaldi) library, which can be pre-installed or installed using our installation script.

```shell
cd tools
# If you have installed Kaldi
KALDI=/your/path/to/Kaldi ./install/install-delta.sh full [cpu|gpu]
# If you have not installed Kaldi, use the following command
# ./install/install-delta.sh full [cpu|gpu]
```

To verify the installation, run:

```shell
# Activate conda environment
conda activate delta-py3.6-tf2.3.0
# Or use the following command if your conda version is < 4.6
# source activate delta-py3.6-tf2.3.0

# Add DELTA environment
source env.sh

# Generate mock data for text classification.
pushd egs/mock_text_cls_data/text_cls/v1
./run.sh
popd

# Train the model
python3 delta/main.py --cmd train_and_eval --config egs/mock_text_cls_data/text_cls/v1/config/han-cls.yml
```

### Manual installation

For advanced installation, full version users, or more details, please refer to [manual installation](docs/installation/manual_setup.md).

### Install from Docker

For Docker users, we provide images with DELTA installed. Please refer to [docker installation](docs/installation/using_docker.md).


## Quick Start

### Existing Examples

DELTA organizes many commonly-used tasks as examples in [egs](egs/) directory. 
Each example is a NLP or speech task using a public dataset.
We provide the whole pipeline including data processing, model training, evaluation, and deployment.

You can simply use the `run.sh` under each directory to prepare the dataset, and then train or evaluate a model. 
For example, you can use the following command to download the CONLL2003 dataset and train and evaluate a BLSTM-CRF model for NER: 

```shell
pushd ./egs/conll2003/seq_label/v1/
./run.sh
popd
python3 delta/main.py --cmd train --config egs/conll2003/seq_label/v1/config/seq-label.yml
python3 delta/main.py --cmd eval --config egs/conll2003/seq_label/v1/config/seq-label.yml
```

### Modeling

There are several modes to start a DELTA pipeline:

- train_and_eval
- train
- eval
- infer
- export_model

**Note**: Before run any command, please make sure you need to `source env.sh` in the current command prompt or a shell script.

You can use `train_and_eval` to start the model training and evaluation:

```shell
python3 delta/main.py --cmd train_and_eval --config <your configuration file>.yml
```

This is equivalent to:

```shell
python3 delta/main.py --cmd train --config <your configuration file>.yml 
python3 delta/main.py --cmd eval --config <your configuration file>.yml 
```

For evaluation, you need to prepare a data file with features and labels. 
If you only want to do inference with feature only, you can use the `infer` mode:

```shell
python3 delta/main.py --cmd infer --config <your configuration file>.yml 
```

When the training is done, you can export a model `checkpoint` to `SavedModel`:

```shell
python3 delta/main.py --cmd export_model --config <your configuration file>.yml 
```

### Deployment

For model deployment, we provide many tools in the DELTA-NN package.
We organize the model deployment scripts under `./dpl` directory.

* Docker pull `zh794390558/delta:deltann-cpu-py3` image, we test deployment under this env.
* Download third-party pacakges by `cd tools && make deltann`.
* Put `SavedModel` and configure `model.yaml` into `dpl/model`.
* Use scripts under `dpl/run.sh` to convert model to other deployment model, and compile libraries.
* All compiled `tensorflow` libs and `delta-nn` libs are in `dpl/lib`.
* All things need for deployment are under `dpl/output` dir.
* Test, benchmark or serve under docker.

For more information, please see [dpl/README.md](dpl/README.md).

### Graph Compiler

Graph Compiler using `TensorFlow grappler`, see [gcompiler/README.md](gcompiler/README.md).

## Benchmarks

In DELTA, we provide experimental results for each task on public datasets as benchmarks. 
For each task, we compare our implementation with a similar model chosen from a highly-cited publication.
You can reproduce the experimental results using the scripts and configuration in the `./egs` directory. 
For more details, please refer to [released models](docs/released_models.md).

### NLP tasks

| Task | Model | DataSet | Metric | DELTA | Baseline | Baseline reference |
|---|---|---|---|---|---|---|
| Sentence Classification | CNN | TREC | Acc | 92.2 | 91.2 | Kim (2014) |
| Document Classification | HAN | Yahoo Answer | Acc | 75.1 | 75.8 | Yang et al. (2016) |
| Named Entity Recognition | BiLSTM-CRF | CoNLL 2003 | F1 | 84.6 | 84.7 | Huang et al. (2015) |
| Intent Detection (joint) | BiLSTM-CRF-Attention | ATIS | Acc | 97.4 | 98.2| Liu and Lane (2016) |
| Slots Filling (joint) | BiLSTM-CRF-Attention | ATIS  | F1 | 95.2 | 95.9 | Liu and Lane (2016) |
| Natural Language Inference | LSTM | SNLI | Acc | 80.7 | 80.6 | Bowman et al. (2016) | 
| Summarization | Seq2seq-LSTM | CNN/Daily Mail | RougeL | 27.3 | 28.1 | See et al. (2017) |
| Pretrain-NER | ELMO | CoNLL 2003 | F1 | 92.2 | 92.2 | Peters et al. (2018) |
| Pretrain-NER | BERT | CoNLL 2003 | F1 | 94.6 | 94.9 | Devlin et al. (2019) |

### Speech tasks

| Task | Model | DataSet | Metric | DELTA | Baseline | Baseline reference |
|---|---|---|---|---|---|---|
| Speech recognition | CTC | HKUST | CER | 36.49 | 38.67 | Miao et al. (2016) |
| Speaker verfication | TDNN | VoxCeleb | EER | 3.028 | 3.138 | Kaldi |
| Emotion recognition | RNN-mean pool | IEMOCAP | Acc | 59.44 | 56.90 | Mirsamadi et al. (2017) |


## FAQ

See [FAQ](docs/faq.md) for more information.

## Contributing

Any contribution is welcome. All issues and pull requests are highly appreciated!
For more details, please refer to [the contribution guide](CONTRIBUTING.md).

## References

Please cite this [paper](https://arxiv.org/abs/1908.01853) when referencing DELTA.
```bibtex
@ARTICLE{delta,
       author = {{Han}, Kun and {Chen}, Junwen and {Zhang}, Hui and {Xu}, Haiyang and
         {Peng}, Yiping and {Wang}, Yun and {Ding}, Ning and {Deng}, Hui and
         {Gao}, Yonghu and {Guo}, Tingwei and {Zhang}, Yi and {He}, Yahao and
         {Ma}, Baochang and {Zhou}, Yulong and {Zhang}, Kangli and {Liu}, Chao and
         {Lyu}, Ying and {Wang}, Chenxi and {Gong}, Cheng and {Wang}, Yunbo and
         {Zou}, Wei and {Song}, Hui and {Li}, Xiangang},
       title = "{DELTA: A DEep learning based Language Technology plAtform}",
       journal = {arXiv e-prints},
       year = "2019",
       url = {https://arxiv.org/abs/1908.01853},
}

```

## License

The DELTA platform is licensed under the terms of the Apache license.
See [LICENSE](LICENSE) for more information.

## Acknowledgement

The DELTA platform depends on many open source repos.
See [References](docs/references.md) for more information.

