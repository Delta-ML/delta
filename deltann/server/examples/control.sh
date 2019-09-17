#!/bin/bash
workspace=$(cd $(dirname $0) && pwd -P)
cd $workspace
app=delta-service
conf=""
action=$1
case $action in
"start" )
    exec "./$app"  -profile "online"
    ;;
* )
    echo "unknow command"
    ;;
esac
