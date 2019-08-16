
# A Emotion Classification Usage Example

In this tutorial, we demonstrate a emotion classification task with an open source dataset: `iemocap`.

A complete process contains following steps:

- Prepare the data set.
- Run egs/iemocap/emo/v1/run.sh script

Before doing any these steps, please make sure that `delta` has been successfully installed. 

Every time you re-open a terminal, don't forget: 

```
source env.sh
```

## Prepare the Data Set

Download iemocap https://sail.usc.edu/iemocap/index.html

## Run

First:

```
cd egs/iemocap/emo/v1
```

Then run run.sh script

```
./run.sh --config_file=conf/emo-keras-blstm.yml --iemocap_root=/iemocap/path
```
