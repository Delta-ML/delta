# DELTA - a DEep Language Technology plAtform 

**DELTA** is a deep learning based end-to-end natural language and speech processing platform. DELTA aims to provide easy and fast experiences for using, deploying, and developing natural language processing and speech models for both academia and industry use cases. DELTA is mainly implemented using TensorFlow.

Currently, DELTA supports many NLP and speech tasks, including text classification, named entity recognition, natural language inference, question and answering, sequence-to-sequence text generation, speech recognition, speaker verification, speech emotion recognition, etc. DELTA has been used for developing several state-of-the-art algorithms for publications and delivering real production to serve millions of users.

For details of DELTA, please refer to this [paper(TBA)](https://arxiv.org/).

**Featuring**:
 - easy-to-use
 - easy-to-deploy
 - easy-to-develop

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Released Models](#released-models)
- [Experiments and Benchmarks](#experiments-and-benchmarks)
- [License](#license)
- [References](#references)

## Installation

### Quick Install

To quickly install DELTA, simply use the following commands:

```shell
# Run the installation script, with NLP-only version or full version and CPU or GPU
# `full` will install `kaldi`, which is usefull for `speech` user
cd tools
./install/install-delta.sh [nlp|full] [cpu|gpu]

# Activate conda environment
conda activate delta-py3.6-tf1.14

# Or
# source activate delta-py3.6-tf1.14
```

This command will create a conda environment with Tensorflow and other required packages.

The full version of delta depends on [Kaldi](https://github.com/kaldi-asr/kaldi). We recommend our users to install Kaldi by following the documents of its official website. With a pre-installed Kaldi, you can install the full version of DELTA by:

```
KALDI=/your/path/to/Kaldi ./install/install-delta.sh full [cpu|gpu]
``` 

To verify the installation, run:

```shell
# Add DELTA enviornment
source env.sh

# generate mock data for text classification.
cd egs/mock_text_cls_data/text_cls/v1
./run.sh
cd -
python3 delta/main.py --cmd train_and_eval --config egs/mock_text_cls_data/text_cls/v1/config/han-cls.yml
```

For advanced installation, `Kaldi` user, or more details, please refer to [manual installation](docs/installation/manual_setup.md).

For Docker users, we provide images with DELTA installed. Please refer to [docker installation](docs/installation/using_docker.md).

## Quick Start

### Data Prepare

DELTA organized data preparation under `egs` directory representing **examples**.
The data-set related pre-processing scripts are under it's own directory.
Run the `run.sh` under this directory, we would have our training data prepared.
The configuration directory are also under this directory.  

### General Commands

The supported commands are listed blow:

- train_and_eval
- train
- eval
- infer
- export_model

Before run any command, please make sure you have source the `env.sh` in the current command prompt or a shell script.

We use only one command to start model train:

```shell
python3 delta/main.py --cmd train_and_eval --config <your configuration file>.yml
```

or the equivalent:

```shell
python3 delta/main.py --cmd train --config <your configuration file>.yml 
python3 delta/main.py --cmd eval --config <your configuration file>.yml 
```

To make an inference:

```shell
python3 delta/main.py --cmd infer --config <your configuration file>.yml 
```

To export a model `checkpoint` to `SavedModel`:

```shell
python3 delta/main.py --cmd export_model --config <your configuration file>.yml 
```

You can also organize all these commands in `run.sh` under your own data-set directory.

### Deployment

DELTA-NN organized model deployment under `dpl` directory.

* Putting `SavedModel` and configure `model.yaml` into `dpl/model`.
* Using script under `dpl/gadapter` to convert model to other deployment model.  
* All compiled `tensorflow` lib and `delta-nn` lib are in `dpl/lib`.
* Testing, benchmarking or serving under docker.

## Released Models

For more details, please refer to [released models](docs/released_models.md).

### Models and Benchmarks

NLP tasks

| Task | Model | DataSet | Metric | DELTA | Baseline | Baseline reference |
|---|---|---|---|---|---|---|
| Sentence Classification | CNN | TREC | Acc | 92.2 | 91.2 | Kim (2014) |
| Document Classification | HAN | Yahoo Answer | Acc | 75.1 | 75.8 | Yang et al. (2016) |
| Named Entity Recognition | BiLSTM-CRF | CoNLL 2003 | F1 | 84.6 | 84.7 | Huang et al. (2015) |
| Intent Detection (joint) | BiLSTM-CRF-Attention | ATIS | Acc | 97.4 | 98.2| Liu and Lane (2016) |
| Slots Filling (joint) | BiLSTM-CRF-Attention | ATIS  | F1 | 95.2 | 95.9 | Liu and Lane (2016) |
| Natural Language Inference | LSTM | SNLI | Acc | 80.7 | 80.6 | Bowman et al. (2016) | 
| Summarization | Seq2seq-LSTM |  |  |  |  |
| Translation | Seq2seq-transformer |  |  |  |  |
| Pretrain-NER | ELMO | CoNLL 2003 | F1 | 92.19 | 92.22 | Matthew et al. (2018) |
| Pretrain-NER | BERT |  |  |  |  |

Speech tasks

| Task | Model | DataSet | Metric | DELTA | Baseline | Baseline reference |
|---|---|---|---|---|---|---|
| Speech recognition | CTC | HKUST | CER |  |  | Mial et al. (2016) |
| Speech recognition | Seq2seq | HKUST | CER |  |  |  |
| Speaker verfication | X-Vector | VoxCeleb | EER |  |  |  |
| Emotion recogniation | ResNet50 | IEMOCAP | Acc | 54.33 | 55.38 | Neumann and Vu (2017) |


## TODO

For more details, please refer to [TODO list](docs/todo.md).

## Contributing

Any kind of contribution is welcome. Please just open an issue without any hesitation.
For more details, please refer to [the contribution guide](CONTRIBUTING.md).

## License

The DELTA platform is licensed under the terms of the Apache license.
See [LICENSE](LICENSE) for more information.

## Acknowlegement

The DELTA platform depends on many open source repos.
See [References](docs/references.rst) for more information.

