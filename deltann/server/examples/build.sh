#!/bin/sh
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
cp -R  ../dpl output
cp $delta_file output/$delta_file
cp control.sh output/$delta_file
cp run.sh output/$delta_file
cp env.sh output/$delta_file


