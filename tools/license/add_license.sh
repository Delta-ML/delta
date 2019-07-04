#!/bin/bash

if [ -z $MAIN_ROOT ];then
  source env.sh
fi

on_exit() {
  unlink $MAIN_ROOT/tools/license/license.cpp.list
  unlink $MAIN_ROOT/tools/license/license.py.list
}

#cpp
if ! [ -f $MAIN_ROOT/tools/license/license.cpp.list ];then
  for dir in delta dpl tools/test utils;
  do
    find $dir -name *.cc -or -name *.h >> $MAIN_ROOT/tools/license/license.cpp.list
  done
  trap on_exit EXIT
fi

while read file;
do
    if ! grep -q Copyright $file
    then
        echo "process: $file"
        cat $MAIN_ROOT/tools/license/LICENSE_cpp $file >$file.new && mv $file.new $file
    fi
done < $MAIN_ROOT/tools/license/license.cpp.list

#python
if ! [ -f $MAIN_ROOT/tools/license/license.py.list ];then
  for dir in delta dpl tools/test utils;
  do
    find $dir -name *.py >> $MAIN_ROOT/tools/license/license.py.list
  done
  trap on_exit EXIT
fi

while read file;
do
    if ! grep -q Copyright $file
    then
        echo "process: $file"
        cat $MAIN_ROOT/tools/license/LICENSE_py $file >$file.new && mv $file.new $file
    fi
done < $MAIN_ROOT/tools/license/license.py.list
