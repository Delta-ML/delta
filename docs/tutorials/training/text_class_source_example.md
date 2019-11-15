# A Text Classification Usage Example

## Intro

In this tutorial, we demonstrate a text classification task with an
open source dataset: `yahoo answer` for users with installation from
source code..

A complete process contains following steps:

- Prepare the data set.
- Set the config file.
- Train a model.
- Export a model
- Deploy the model.

Before doing any these steps, please make sure that `delta` has been successfully installed. 

Every time you re-open a terminal, don't forget: 

```
source env.sh
```

## Prepare the Data Set

You can refer to directory: `egs` for data preparing. In our example, `egs/yahoo_answer` contains data preparing including downloading and reformat. 

First:

```
cd egs/yahoo_answer/text_cls/v1
```
Then run the script: 

```
./run.sh
```

The generated data are in directory: `data/yahoo_answer`. 

The generated data for text classification should be in the standard format for text classification, which is "label\tdocument".

## Set the Config File

The config file of this example is `egs/yahoo_answer/text_cls/v1/config/cnn-cls.yml`

In the config file, we set the task to be `TextClsTask` and the model to be `HierarchicalAttentionModel`.

### Config Details

The config is composed by 3 parts: `data`, `model`, `solver`.
 
Data related configs are under `data`. You can set the data path (including training set, dev set and test set). The data process configs can also be found here (mainly under `task`). For example, we set `use_dense: false` since no dense input was used here. We set `language: chinese` since it's a Chinese text. 

Model parameters are under `model`. The most important config here is `name: SeqclassCNNModel`, which specifies the model to use. Detail structure configs are under `net->structure`. Here, the `filter_sizes` are 3, 4, 5 and `num_filters` is 128.

The configs under `solver` are used by solver class, including training optimizer, evaluation metrics and checkpoint saver. Here the class is `RawSolver`.

## Train a Model

After setting the config file, you are ready to train a model.

```
python delta/main.py --cmd train_and_eval --config egs/yahoo_answer/text_cls/v1/config/cnn-cls.yml
```

The argument `cmd` tells the platform to train a model and also evaluate the dev set during the training process.

After enough steps of training, you would find the model checkpoints have been saved to the directory set by `saver->model_path`, which is `exp/yahoo_answer/ckpt/cnn-cls` in this case.

## Export a Model

If you would like to export a specific checkpoint to be exported, please set `infer_model_path` in config file. Otherwise, platform will simply find the newest checkpoint under the directory set by `saver->model_path`.

```
python delta/main.py --cmd export_model --config egs/yahoo_answer/text_cls/v1/config/cnn-cls.yml
```

The exported models are in the directory set by config `service->model_path`, which is `exp/yahoo_answer/cnn-cls/service` here.

## Deploy the Model

Before model deploying, please make sure that `deltann` has been successfully installed.
