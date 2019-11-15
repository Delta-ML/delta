# DELTA - A DEep learning Language Technology plAtform

## What is DELTA?

**DELTA** is a deep learning based end-to-end **natural language and speech processing** platform. 
DELTA aims to provide easy and fast experiences for using, deploying, and developing natural language processing and speech models for both academia and industry use cases. DELTA is mainly implemented using TensorFlow and Python 3.

For details of DELTA, please refer to this [paper](https://arxiv.org/abs/1908.01853).

## What can DELTA do?

DELTA has been used for developing several state-of-the-art algorithms for publications and delivering real production to serve millions of users. 
It helps you to train, develop, and deploy NLP and/or speech models, featuring:

 - Easy-to-use
   - One command to train NLP and speech models, including:
     - NLP: text classification, named entity recognition, question and answering, text summarization, etc
     - Speech: speech recognition, speaker verification, emotion recognition, etc
   - Use configuration files to easily tune parameters and network structures
 - Easy-to-deploy
   - What you see in training is what you get in serving: all data processing and features extraction are integrated into a model graph
   - Uniform I/O interfaces and no changes for new models
 - Easy-to-develop
   - Easily build state-of-the-art models using modularized components
   - All modules are reliable and fully-tested

## References

Please cite this [paper](https://arxiv.org/abs/1908.01853) when referencing DELTA.
```bibtex
@ARTICLE{delta,
       author = {{Han}, Kun and {Chen}, Junwen and {Zhang}, Hui and {Xu}, Haiyang and
         {Peng}, Yiping and {Wang}, Yun and {Ding}, Ning and {Deng}, Hui and
         {Gao}, Yonghu and {Guo}, Tingwei and {Zhang}, Yi and {He}, Yahao and
         {Ma}, Baochang and {Zhou}, Yulong and {Zhang}, Kangli and {Liu}, Chao and
         {Lyu}, Ying and {Wang}, Chenxi and {Gong}, Cheng and {Wang}, Yunbo and
         {Zou}, Wei and {Song}, Hui and {Li}, Xiangang},
       title = "{DELTA: A DEep learning based Language Technology plAtform}",
       journal = {arXiv e-prints},
       year = "2019",
       url = {https://arxiv.org/abs/1908.01853},
}
