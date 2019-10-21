#! /bin/bash
set -euf -o pipefail

delta_file="delta-service"

if [ -f "$delta_file" ]; then
rm -rf $delta_file
fi

if [ -d output ]
then
    rm -rf output
fi

mkdir -p output/$delta_file/log
#cp -R  ../dpl output
mkdir -p output/dpl/output/include
cp -R  ../../../dpl/lib/ output/dpl/output

pushd output/dpl/output/lib/custom_ops
mv x_ops.so libx_ops.so
popd

cp -R  ../../../dpl/model/ output/dpl/output
mkdir -p output/dpl/output/model/saved_model/1

pushd output/dpl/output/model/saved_model/
mv  saved_model.pbtxt variables  1/
popd

cp -R ../../api/c_api.h output/dpl/output/include

cp -R output/dpl ../

GOOS=linux GOARCH=amd64 go build -o $delta_file  main.go

cp $delta_file output/$delta_file
cp control.sh output/$delta_file
cp run.sh output/$delta_file
cp server-env.sh output/$delta_file


