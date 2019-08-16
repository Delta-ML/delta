
# A Emotion Classification Usage Example

In this tutorial, we demonstrate a emotion classification task with an open source dataset: `iemocap`.

A complete process contains following steps:

- Prepare the data set.
- Set the run.sh script.
- Run

Before doing any these steps, please make sure that `delta` has been successfully installed. 

Every time you re-open a terminal, don't forget: 

```
source env.sh
```

## Prepare the Data Set

Iemocap https://sail.usc.edu/iemocap/index.html

## Set run.sh script

The run.sh script in `egs/iemocap/emo/v1`

Set config file

The config files of this example in `egs/iemocap/emo/v1/conf`

```
config_file=conf/emo-keras-blstm.yml
```

Set the data path

```
iemocap_root=/iemocap/path
```

## Run

```
./run.sh
```
