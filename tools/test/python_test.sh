#!/bin/bash

if [ -z $MAIN_ROOT ];then
  if [ -f env.sh ];then 
      source env.sh
  else
      source ../../env.sh
  fi
fi

# generate mock data
bash $MAIN_ROOT/tools/test/gen_mock_egs_data.sh

# python unist test
tmpfile=`mktemp /tmp/python_test.XXXXXX`

find $MAIN_ROOT/delta -name *_test.py  &> $tmpfile

retcode=0
while read line
do
  echo Testing $line
  python3 $line
  if [ $? != 0 ];then
      echo Test $line error. > /dev/stderr
      retcode=1
  fi
done < $tmpfile

if [ $retcode != 0 ]; then
  exit $retcode
fi

# integration test
bash $MAIN_ROOT/tools/test/integration_test.sh

