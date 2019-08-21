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

 | Model | Dataset | EER(%) | minDCF 0.01 | minDCF 0.001 | Baseline(EER/minDCF0.01/minDCF0.001) | Reference| Config |
 | ---   | ---  |  --- | ---  | --- | --- | --- | --- |
 | TDNN+Softmax+Consine backend | voxceleb | 5.795 | - | - | - | kaldi | conf/tdnn_softmax.yml |
 | TDNN+Softmax+PLDA backend| voxceleb | 3.097 | 0.3405 | 0.4888 | 3.128/0.3258/0.5003 | kaldi | conf/tdnn_softmax.yml |

