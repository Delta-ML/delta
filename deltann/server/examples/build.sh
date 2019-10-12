#! /bin/bash
set -euf -o pipefail

delta_file="delta-service"
if [ -f "$delta_file" ]; then
rm -rf $delta_file
fi

GOOS=linux GOARCH=amd64 go build -o $delta_file  main.go
if [ -d output ]
then
    rm -rf output
fi
mkdir -p output/$delta_file/log
#cp -R  ../dpl output
mkdir -p output/dpl/output/include
cp -R  ../../../dpl/lib/ output/dpl/output
mv output/dpl/output/lib/custom_ops/x_ops.so libx_ops.so
cp -R  ../../../dpl/model/ output/dpl/output
cp -R ../../api/c_api.h output/dpl/output/include
cp $delta_file output/$delta_file
cp control.sh output/$delta_file
cp run.sh output/$delta_file
cp env.sh output/$delta_file



