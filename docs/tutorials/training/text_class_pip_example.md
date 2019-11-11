# A Text Classification Usage Example for pip users

## Intro

In this tutorial, we demonstrate a text classification task with a
demo mock dataset **for users install by pip**.

A complete process contains following steps:

- Prepare the data set.
- Develop custom modules (optional).
- Set the config file.
- Train a model.
- Export a model

Please clone our demo repository:

```bash
git clone --depth 1 https://github.com/applenob/delta_demo.git
cd ./delta_demo
```

## A quick review for installation

If you haven't install `delta-nlp`, please:

```bash
pip install delta-nlp
```

**Requirements**: You need `tensorflow==2.0.0` and `python==3.6` in
MacOS or Linux.

## Prepare the Data Set

run the script: 

```
./gen_data.sh
```

The generated data are in directory: `data`. 

The generated data for text classification should be in the standard format for text classification, which is "label\tdocument".

## Develop custom modules (optional)

Please make sure we don't have modules you need before you decide to
develop your own modules.

```python
@registers.model.register
class TestHierarchicalAttentionModel(HierarchicalModel):
  """Hierarchical text classification model with attention."""

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)

    logging.info("Initialize HierarchicalAttentionModel...")

    self.vocab_size = config['data']['vocab_size']
    self.num_classes = config['data']['task']['classes']['num_classes']
    self.use_true_length = config['model'].get('use_true_length', False)
    if self.use_true_length:
      self.split_token = config['data']['split_token']
    self.padding_token = utils.PAD_IDX
```

You need to register this module file path in the config file
`config/han-cls.yml` (relative to the current work directory).

```yml
custom_modules:
  - "test_model.py"
```

## Set the Config File

The config file of this example is `config/han-cls.yml`

In the config file, we set the task to be `TextClsTask` and the model to be `TestHierarchicalAttentionModel`.

### Config Details

The config is composed by 3 parts: `data`, `model`, `solver`.
 
Data related configs are under `data`. 
You can set the data path (including training set, dev set and test set). 
The data process configs can also be found here (mainly under `task`). 
For example, we set `use_dense: false` since no dense input was used here. 
We set `language: chinese` since it's a Chinese text. 

Model parameters are under `model`. The most important config here is
`name: TestHierarchicalAttentionModel`, which specifies the model to
use. Detail structure configs are under `net->structure`. Here, the 
`max_sen_len` is 32 and `max_doc_len` is 32.

The configs under `solver` are used by solver class, including training optimizer, evaluation metrics and checkpoint saver. 
Here the class is `RawSolver`.

## Train a Model

After setting the config file, you are ready to train a model.

```
delta --cmd train_and_eval --config config/han-cls.yml
```

The argument `cmd` tells the platform to train a model and also evaluate
the dev set during the training process.

After enough steps of training, you would find the model checkpoints have been saved to the directory set by `saver->model_path`, which is `exp/han-cls/ckpt` in this case.

## Export a Model

If you would like to export a specific checkpoint to be exported, please set `infer_model_path` in config file. Otherwise, platform will simply find the newest checkpoint under the directory set by `saver->model_path`.

```
delta --cmd export_model --config/han-cls.yml
```

The exported models are in the directory set by config
`service->model_path`, which is `exp/han-cls/service` here.

