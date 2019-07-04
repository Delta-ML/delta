# Emotion

using for conflict detection

# data description 

- data/speech_cls_tas.py, data input pipeline, support `wav` and `feat` data 
  data dir should like below:
  ```text
  data/
     conflict/
       id.wav
       id.TextGrid
       ...
     normal/
       id.wav
       ...
     textgrid.segments
  ```

  `textgrid.segments` generate by `util/gen_segments.sh`
  ```text
  /datasets/data/emotion/train/conflict/985e92a1636b73fec794fe6_20180907.wav (0.000000,35.611097) (79.816283,219.723946) (227.054141,300.340000)
  /datasets/data/emotion/train/conflict/4e110127b04c0d3d269b6df_20180907.wav (0.000000,18.883552) (127.398159,203.824681) (246.327001,268.460183)
  ```

# other

- eval_pb.py, eval with graph.pb
- eval_simple_save.py, eval with savedModel
- gen_feat.py, generate feat for wav, used by `feat` mode
- python_speech_features, fbank, mfcc, delta_detlas src
- tf_speech_feature.py, extract feature in TF graph, used by `wav` mode
