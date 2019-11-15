# Reproduce experiments - egs 

The `egs` director is data-oriented for `data prepration` and model `training`, `evaluation` and `infering`.

Sppech and NLP task are orgnized by `egs`, e.g. ASR, speaker verfication, NLP.

## An Egs Example

In this tutorial, we demonstrate an emotion recognition task with an open source dataset: `IEMOCAP`. All other task is same to this.

A complete process contains following steps:

- Download the `IEMOCAP` corpus.
- Run egs/iemocap/emo/v1/run.sh script

Before doing any these steps, please make sure that `delta` has been successfully installed. 

Every time you re-open a terminal, don't forget: 

```
source env.sh
```

### Prepare the Data Set

Download `IEMOCAP` from https://sail.usc.edu/iemocap/index.html

### Run

First:

```
pushd egs/iemocap/emo/v1
```

Then run `run.sh` script

```
./run.sh --iemocap_root=</path/to/iemocap>
```

For other task, e.g. `ASR`, `Speaker`, the main script is `run_delta.sh`, but default main root is `run.sh`.


