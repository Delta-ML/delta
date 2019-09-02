 # Voxceleb Recipe

 This recipe replaces ivectors used in the v1 recipe with embeddings extracted
 from a deep neural network.  In the scripts, we refer to these embeddings as
 "xvectors."  The recipe is closely based on the following paper:
 http://www.danielpovey.com/files/2018_icassp_xvectors.pdf but uses a wideband
 rather than narrowband MFCC config.

 In addition to the VoxCeleb datasets used for training and evaluation (see
 ../../README.txt) we also use the following datasets for augmentation.

     MUSAN               http://www.openslr.org/17
     RIR_NOISES          http://www.openslr.org/28

## Results

 | Model | Dataset | EER(%) | minDCF 0.01 | minDCF 0.001 | Baseline(EER/minDCF0.01/minDCF0.001) | Reference | Config | NGPU | Front End |
 | ---   | ---  |  --- | ---  | --- | --- | --- | --- | --- | --- |
 | ivector + PLDA backend | voxceleb | 5.329 | 0.4933 | 0.6168 | - | kaldi | - | - | kaldi |
 | TDNN + Softmax + PLDA backend | voxceleb | 3.128 | 0.3258 | 0.5003 | - | kaldi | - | - | kaldi |
 | ---   | ---  |  --- | ---  | --- | --- | --- | --- | --- | --- |
 | TDNN + Softmax + Consine backend | voxceleb | 5.795 | - | - | - | kaldi | conf/tdnn_softmax.yml | 1 | kaldi |
 | TDNN + Softmax + PLDA backend| voxceleb | 3.028 | 0.3101 | 0.4875 | 3.128/0.3258/0.5003 | kaldi | conf/tdnn_softmax.yml | 1 | kaldi |
 | ---   | ---  |  --- | ---  | --- | --- | --- | --- | --- | --- |
 | TDNN + Arcface + Consine backend | voxceleb | 4.4538 | - | - | - | - | conf/tdnn_arcface.yml | 1 | kaldi |
 | TDNN + Arcface + PLDA backend | voxceleb | 2.789 | 0.3152 | 0.5391 | - | - | conf/tdnn_arcface.yml | 1 | kaldi |

