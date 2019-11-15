#!/bin/bash

if [ $# != 2 ];then
  echo "usage: $0 [memcheck|massif] [program_bin]"
  exit 1
fi

set -e

TOOL=$1
BIN=$2

if [ $(valgrind --version > /dev/null) != 0 ];then
  apt-get install valgrind
fi

if [ $TOOL == "memcheck" ];then
  valgrind --leak-check=full --log-file=memcheck.log --show-leak-kinds=all $BIN 
elif [ $TOOL == "massif" ];then
  valgrind --tool=massif --log-file=massif.log $BIN 
  ms_print massif.log > massif.txt
fi
