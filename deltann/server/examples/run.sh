#! /bin/bash
set -euf -o pipefail
nohup ./delta-service -log_dir=./log -alsologtostderr=true  -port "8004" -yaml "../dpl/output/model/saved_model/1/model.yaml"  -type "predict"  -debug true &
