# Released Models

## NLP Models

### Sequence classification

We provide the sequence classification models using CNN, LSTM, HAN (hierarchical attention networks), transformer, etc.

## Sequence labeling

We provide the LSTM based sequence labeling and an LSTM with CRF based method.

## Pairwise modeling

We implement the match of text pairwise models computing similarity across sentence representation encoded with two LSTM.


## Sequence-to-sequence (seq2seq) modeling

We implement the standard seq2seq models using LSTM with attention and transformers. Note that, the seq2seq structure is also used for speech recognition. In DELTA, this part is shared between NLP and ASR tasks.

## Multi-task modeling

We implement a multi-task model for sequence classification and labeling, where the sequence level loss and the step level loss are computed simultaneously. This model is used to jointly train an intent recognizer and named entity recognizer together.

## Pretraining integration

We implement an interface to integrate a pretrained model into a DELTA model, where the pretrained model is used to dynamically generate embedding which is concatenated with the word embedding for the different task. To be specific, a user can pretrain an ELMO or BERT model first and then build a DELTA model with the prertained model. Both model will be combined into a TensorFlow graph for training and inference. The ELMO or BERT models trained from the official open-sourced libraries can be directly used in DELTA. 

## Speech models

### Automatic speech recognition (ASR)

We provide an attention based seq2seq ASR model. We also implement another popular type of ASR model using connectionist temporal classification (CTC).

### Speaker Verification/Identification

We provide an X-vector text-independent model and an end-to-end model.

### Speech emotion recognition

Recently several deep learning based approaches have been successfully used in speech emotion recognition and we implement some models the in DELTA.

## Multimodal models

### Textual+acoustic

 In our implementation, we use two sequential models (e.g.,CNNs or LSTMs) to learn the sequence embedding for speech and text separately, and thenconcatenates the learned embedding vectors for classification.

### Textual+numeric

We implement the direct concatenation data fusion in data processing stage,therefore this type of multimodal training can be directly used for existing models in DELTA.



