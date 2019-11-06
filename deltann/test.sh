#!/bin/bash
if [ $# != 1 ];then
  echo "usage: $0 [speaker|text_cls|dir_name_under_examples]"
  exit 1
fi

set -e

make examples

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/../dpl/lib/tensorflow:$PWD/../dpl/lib/deltann/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/../dpl/lib/tensorflow:$PWD/.gen/lib


case $1 in
 text_cls)
   echo "text_cls"
   ./examples/text_cls/test.bin examples/text_cls/model.yaml
   ;;
 speaker)
   echo "speaker"
   ./examples/speaker/test.bin examples/speaker/model.yaml
   ;;
 *)
   echo "Error param: $1"
   exit 1
   ;;
esac

exit 0
