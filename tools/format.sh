#!/bin/bash
set -e

if [ $(id -u) == 0 ]; then
  SUDO=
else
  SUDO=sudo
fi
parallel --version &> /dev/null || ${SUDO} apt-get install -y parallel

PYTEMPFILE=`mktemp`
trap 'unlink $PYTEMPFILE' EXIT INT QUIT ABRT

# yapf
yapf -version &> /dev/null || ${SUDO} pip install yapf

# yapf
for dir in delta deltann dpl docker utils;
do
  find $dir -name *.py >> $PYTEMPFILE
done
#find tools \( -path tools/tensorflow \
#    -o -path tools/abseil-cpp \
#    -o -path tools/cppjieba \
#    -o -path tools/kaldi \
#    -o -path tools/googletest \
#    -o -path tools/espnet \) \
#    -prune \
#    -name *.py >> $PYTEMPFILE

find tools/test \
   tools/license \
   tools/plugins \
   -name *.py >> $PYTEMPFILE
cat $PYTEMPFILE | parallel -I{} 'yapf -i {}; echo "yapf: {}";'



clang-format -version &> /dev/null || ${SUDO} apt-get install -y clang-format

CPPTEMPFILE=`mktemp`
trap 'unlink $CPPTEMPFILE' EXIT INT QUIT ABRT

# clang-format
for dir in delta deltann dpl docker utils;
do
  find $dir -name '*.h' -o -name '*.c' -o -name '*.cpp' -o -name '*.cc' >> $CPPTEMPFILE
done
#find tools \( -path tools/abseil-cpp \
#    -o -path tools/cppjieba \
#    -o -path tools/tensorflow \
#    -o -path tools/kaldi \
#    -o -path tools/googletest \
#    -o -path tools/kaldi \
#    -o -path tools/espnet \) \
#    -prune \
#    -name '*.h' -o -name '*.c' -o -name '*.cc' -o -name '*.cpp' >> $CPPTEMPFILE

find tools/test \
   tools/license \
   tools/plugins \
   -name '*.h' -o -name '*.c' -o -name '*.cc' -o -name '*.cpp' >> $CPPTEMPFILE
cat $CPPTEMPFILE | parallel -I{} 'clang-format -i {}; echo "clang-format: {}";'
