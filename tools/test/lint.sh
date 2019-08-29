#!/bin/bash

if [ -z $MAIN_ROOT ];then
  if [ -f env.sh ];then 
      source env.sh
  else
      source ../../env.sh
  fi
fi

export PATH=$PATH:$HOME/.local/bin

cpplint --version &> /dev/null || sudo pip install --user cpplint

CPPTMPFILE=`mktemp`

# cpplint
for dir in delta deltann tools/test;
do
    find $MAIN_ROOT/$dir -name '*.c' -o -name '*.cc' -o -name '*.h'  >> $CPPTMPFILE
done

while read file;
do
    echo "cpplint: $file"
    cpplint $file 
done < $CPPTMPFILE

pylint --version &> /dev/null || sudo pip install --user pylint

PYTMPFILE=`mktemp`

# pylint
for dir in delta deltann egs dpl tools/test utils;
do
    find $MAIN_ROOT/$dir -name '*.py' >> $PYTMPFILE
done

while read file;
do
    echo "pylint: $file"
    pylint -j `nproc` $file
done < $PYTMPFILE

on_exit(){
  unlink $CPPTMPFILE
  unlink $PYTMPFILE
}
trap on_exit EXIT ABRT QUIT

#flake8
flake8
