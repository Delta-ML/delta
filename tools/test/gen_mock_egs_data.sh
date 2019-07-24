#!/bin/bash

if [ -z $MAIN_ROOT ];then
  source env.sh
fi

tmpfile=`mktemp /tmp/mock_run.XXXXXX`

find $MAIN_ROOT/egs/mock_* -name run.sh  &> $tmpfile

retcode=0
while read line
do
  echo "Run $line"
  rundir=`dirname "$line"`
  cd $rundir
  bash run.sh
  if [ $? != 0 ];then
      echo "Run $line error." > /dev/stderr
      retcode=1
  fi
done < $tmpfile

cd $MAIN_ROOT
exit $retcode
