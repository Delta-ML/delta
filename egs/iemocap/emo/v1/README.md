This recipe do emotion recognition on IEMOCAP.

## Results
This recipe do emotion recognition on `improvised` and `scripted` speech.

| model | config | data |  acc | baseline | reference| ngpus | front end |
| ---   | ---    | ---  |  --- | ---      |  ---     | --- | --- |
 |align-multimodal|multimodal-align-emotion.yml|all|72.5|72.5|[ (Xu et al. (2019)](https://arxiv.org/pdf/1909.05645.pdf)|1|delta|
| resnet50 | conf/emo-keras-resnet50.ymal | all | 59.15 | 56.10 | [Neumann and Vu (2017)](https://arxiv.org/abs/1706.00612) | 1 | delta |
| rnn-mean pool | conf/emo-keras-blstm.yaml | impro |  65.23 | 56.90 | [Mirsamadi et al. (2017)](https://personal.utdallas.edu/~mirsamadi/files/mirsamadi17a.pdf) | 1 | delta |
| rnn-mean pool | conf/emo-keras-blstm.yaml | all |  59.44 | 56.90 | [Mirsamadi et al. (2017)](https://personal.utdallas.edu/~mirsamadi/files/mirsamadi17a.pdf) | 1 | delta |

## Run
You can use run.sh (the last three models) and run_wwm.sh (align-multimodal) to process data, train and eval.
## Citation
If you find the resource useful, you should cite the following paper:
```
@article{xu2019learning,
  title={Learning Alignment for Multimodal Emotion Recognition from Speech$\}$$\}$},
  author={Xu, Haiyang and Zhang, Hui and Han, Kun and Wang, Yun and Peng, Yiping and Li, Xiangang},
  journal={Proc. Interspeech 2019},
  pages={3569--3573},
  year={2019}
}
@article{han2019delta,
  title={DELTA: A DEep learning based Language Technology plAtform},
  author={Han, Kun and Chen, Junwen and Zhang, Hui and Xu, Haiyang and Peng, Yiping and Wang, Yun and Ding, Ning and Deng, Hui and Gao, Yonghu and Guo, Tingwei and others},
  journal={arXiv preprint arXiv:1908.01853},
  year={2019}
}
```

