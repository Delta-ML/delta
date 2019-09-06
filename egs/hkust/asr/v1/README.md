# HKUST

## Results

Results on `test` subsets.

| Model | Token | Feat | CER% | LM | Decoder | Baseline | Reference | Config | NGPU | Front End |
| --- | --- | --- | --- |  --- | --- | --- | --- | --- | --- | --- |
| 5BLSTM | Charactor | 83 fbank+pitch | 34.89 | w/o | greedy | 38.67 | [Miao et al. (2016)](https://www.cs.cmu.edu/~ymiao/pub/icassp2016_ctc.pdf) | asr-ctc.yml | 2 | kaldi |
| 5BLSTM | Charactor | 43 fbank+pitch | 36.49 | w/o | greedy  | 38.67 | [Miao et al. (2016)](https://www.cs.cmu.edu/~ymiao/pub/icassp2016_ctc.pdf) | - | 2 | kaldi |
| 2CNN+4BLSTM | Charactor | 83 fbank+pitch | 31.55 | w/o | greedy | - | - | CTCAsrModel | 2 | kaldi |
