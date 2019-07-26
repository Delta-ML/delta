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
for dir in $MAIN_ROOT/delta $MAIN_ROOT/tools/test;
do
    find $dir -name '*.c' -o -name '*.cc' -o -name '*.h'  >> $CPPTMPFILE
done

while read file;
do
    echo "cpplint: $file"
    cpplint $file 
done < $CPPTMPFILE

pylint --version &> /dev/null || sudo pip install --user pylint

PYTMPFILE=`mktemp`

# pylint
for dir in $MAIN_ROOT/delta $MAIN_ROOT/tools/test;
do
    find $dir -name '*.py' >> $PYTMPFILE
done

while read file;
do
    echo "pylint: $file"
    pylint -j 1 $file 
done < $PYTMPFILE

on_exit(){
  unlink $CPPTMPFILE
  unlink $PYTMPFILE
}
trap on_exit EXIT ABRT QUIT
