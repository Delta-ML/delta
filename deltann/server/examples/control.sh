#!/bin/bash
workspace=$(cd $(dirname $0) && pwd -P)
cd $workspace
app=delta-service
conf=""
action=$1
case $action in
"start" )
    exec "./$app"  -log_dir=./log -alsologtostderr=true  -port "8004" -yaml "../dpl/output/conf/model.yaml"  -type "classify"
    ;;
* )
    echo "unknow command"
    ;;
esac
