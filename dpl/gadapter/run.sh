#!/bin/bash

# converte SavedModel to TF GraphDef, TF Lite or TensorRT graph

stage=0

model_config='model_config.sh'
saved_model_path='saved_model'

. utils/parse_options.sh $ e.g. this parses the --stage option if supplied.
