#!/bin/bash

tmpfile=`mktemp /tmp/python_test.XXXXXX`

find $MAIN_ROOT/delta -name *_test.py  &> $tmpfile

retcode=0
while read line
do
  echo Testing $line
  python $line
  if [ $? != 0 ];then
      echo Test $line error. > /dev/stderr
      retcode=1
  fi
done < $tmpfile

exit $retcode



