#!/bin/sh
GOOS=linux GOARCH=amd64 go build -o delta-service  main.go
if [ -d output ]
then
    rm -r output
fi
mkdir -p output/log
cp delta-service output
cp control.sh output


