#!/bin/bash

source server-env.sh

workspace=$(cd $(dirname $0) && pwd -P)
cd $workspace

app=delta-service
conf=""

action=$1
case $action in
"start" )
    exec "./$app"  -log_dir=./log -alsologtostderr=true  -port "8004" -yaml "../dpl/output/model/saved_model/1/model.yaml" -type "predict"  -debug true
    ;;
* )
    echo "usage: $0 [start]"
    ;;
esac
