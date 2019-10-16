#! /bin/bash
set -euf -o pipefail
source server-env.sh
nohup ./delta-service -log_dir=./log -alsologtostderr=true  -port "8004" -yaml "../dpl/output/model/model.yaml"  -type "predict"  -debug true &
