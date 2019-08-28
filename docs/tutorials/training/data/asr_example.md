# ASR Data

This tutorials discusses how to deal with automatic speech recognition(ASR) tasks on the basis of DELTA.

## Data descripition

For data preparing, you can refer to directory:'egs/hkust/asr/v1'. 
By simply using `./run.sh`, an open source dataset, HKUST, can be quickly downloaded and reformated like below: 

 ```json
 uttID: {
    "input": [
       {
         "feat": the file and the position while the feats of current utterance is sorted
         "name": "input1"
         "shape" : [
             number_frames
             dimension_feats
         ]
       }
    ],
    "output": [
        {
            "name": "target",
            "shape": [
                number_words
                number_classes
            ],
            "text":
            "token":
            "tokenid":
        }
    ],
    "utt2spk": speaker index
 }
 ```
 It should be noted that num_classes = size_vocabulary + 2， where size_vocabulary is the size of the vocabulary. The zero value and the largest value（num_classes - 1）is reserved for the blank and sos/eos label respectively.
 For Example, the vocabulary is consist of 3 different labels [a, b, c]. Then, num_classes = 5 and the labels indexing is {blank:0, a:1, b:2, c:3, sos/eos:4}

## Model training

1. For ASR tasks, a default config file is written in `conf/asr-ctc.yml`.
  Two different CTC-based model, `CTCAsrModel` and `CTCRefAsrModel`, are supported in DELTA. The details of them can be seen in `delta/models/asr_model.py`. 

2. After setting the config file, the following script can be executed to train a ASR model:

 ```bash
 python3 delta/main.py --config egs/hkust/asr/v1/conf/asr-ctc.yml --cmd train_and_eval 
 ```
