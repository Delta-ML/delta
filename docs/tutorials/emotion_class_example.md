
# A Emotion Classification Usage Example

In this tutorial, we demonstrate a emotion classification task with an open source dataset: `iemocap`.

A complete process contains following steps:

- Prepare the data set.
- Run

Before doing any these steps, please make sure that `delta` has been successfully installed. 

Every time you re-open a terminal, don't forget: 

```
source env.sh
```

## Prepare the Data Set

Iemocap https://sail.usc.edu/iemocap/index.html

## Run

The run.sh script in `egs/iemocap/emo/v1`,  config files in `egs/iemocap/emo/v1/conf`

```
./run.sh --config_file=conf/emo-keras-blstm.yml --iemocap_root=/iemocap/path
```
