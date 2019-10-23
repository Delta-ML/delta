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
mkdir -p output/dpl/

cp -R ../../../dpl/output/ output/dpl/

cp -R ../mock_yaml/nlp/model.yaml   output/dpl/output/model/saved_model/1/

if [ -d ../output ]
then
    rm -rf ../output
fi

cp -R output/dpl ../

GOOS=linux GOARCH=amd64 go build -o $delta_file  main.go

cp $delta_file output/$delta_file
cp control.sh output/$delta_file
cp run.sh output/$delta_file
cp server-env.sh output/$delta_file


