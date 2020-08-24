#!/bin/bash
# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

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

find $MAIN_ROOT -name *_test.py  &> $tmpfile

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

if [ $retcode -ne 0 ]; then
  exit $retcode
fi

# integration test
bash $MAIN_ROOT/tools/test/integration_test.sh

