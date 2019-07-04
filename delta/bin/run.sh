#!/bin/bash

# text cls
#CUDA_VISIBLE_DEVICES='1'  python -u delta/main.py --cmd train --config delta/config/new-cnn-cls/cnn-cls.yml

# data
#CUDA_VISIBLE_DEVICES=0  python data/speech_cls_task.py
# feat
#CUDA_VISIBLE_DEVICES=0  python data/feat/speech_feature.py

# emotion
#CUDA_VISIBLE_DEVICES='3'  python -u delta/main.py --cmd eval --config delta/config/emotion-speech-cls/emotion-speech-cls.yml 
#CUDA_VISIBLE_DEVICES='3'  python -u delta/main.py --cmd export_model --config delta/config/emotion-speech-cls/emotion-speech-cls.yml 
#CUDA_VISIBLE_DEVICES='3'  python -u delta/main.py --cmd infer --config delta/config/emotion-speech-cls/emotion-speech-cls.yml 
#CUDA_VISIBLE_DEVICES='3'  python -u delta/main.py --cmd train --config delta/config/emotion-speech-cls/emotion-speech-cls.yml 
#CUDA_VISIBLE_DEVICES='3'  python -u delta/main.py --cmd gen_cmvn --config delta/config/emotion-speech-cls/emotion-speech-cls.yml
#CUDA_VISIBLE_DEVICES='2'  python -u delta/main.py --cmd gen_feat --config delta/config/emotion-speech-cls/emotion-speech-cls.yml --dry_run True
CUDA_VISIBLE_DEVICES=''  python -u delta/main.py --cmd train_and_eval --config delta/config/emotion-speech-cls/emotion-speech-cls.yml

# kws
#CUDA_VISIBLE_DEVICES='3'  python -u delta/main.py --cmd train_and_eval --config delta/config/kws-cls/kws_speech_cls.yml
#CUDA_VISIBLE_DEVICES='3'  python -u delta/main.py --cmd export_model --config delta/config/kws-cls/kws_speech_cls.yml
#CUDA_VISIBLE_DEVICES='3'  python -u delta/main.py --cmd infer --config delta/config/kws-cls/kws_speech_cls.yml

# metrics test
# python  -u utils/metrics/sklearn_metrics_test.py
# python  -u utils/metrics/metric_utils_test.py 

# serving
# python  -u serving/eval_speech_cls_pb.py --config delta/config/emotion-speech-cls/emotion-speech-cls.yml --gpu None --mode eval

# emotion w/ vad and/or text 
#CUDA_VISIBLE_DEVICES='2'  python -u delta/main.py --cmd train_and_eval --config delta/config/emotion-speech-cls/emotion-vad-speech-cls.yml 

# door
#CUDA_VISIBLE_DEVICES='2'  python -u delta/main.py --cmd train_and_eval --config delta/config/emotion-speech-cls/door-speech-cls.yml 

# iemocap
#CUDA_VISIBLE_DEVICES='2'  python -u delta/main.py --log_debug true --cmd train_and_eval --config delta/config/emotion-speech-cls/iemocap-speech-cls.yml 
#CUDA_VISIBLE_DEVICES='2'  python -u delta/main.py --cmd gen_feat --config delta/config/emotion-speech-cls/iemocap-speech-cls.yml

# espnet pipeline
#:$PWD/tools/espnet python  -u delta/data/utils/*test.py

# asr
#CUDA_VISIBLE_DEVICES='' python -u delta/main.py --cmd train_and_eval --config delta/config/asr-seq/asr-seq.yml
