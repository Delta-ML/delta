#!/bin/bash
# inspect saved model

# https://www.tensorflow.org/beta/guide/saved_model#saved_model_cli
# https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/tutorials/Serving_REST_simple.ipynb

set -e

if [ $# != 1 ];then
  echo "usage: $0 saved_model_dir"
  exit 1
fi

saved_model_dir=$1

echo "inspect model: [$PWD/$saved_model_dir]"

echo "show"
saved_model_cli show --dir $saved_model_dir
echo
echo "show serve"
saved_model_cli show --dir $saved_model_dir  --tag_set serve
echo
echo "show serve serving_default"
saved_model_cli show --dir $saved_model_dir  --tag_set serve --signature_def serving_default
echo
echo "show all"
saved_model_cli show --dir $saved_model_dir  --all
echo
