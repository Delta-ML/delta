#!/bin/bash

PYTEMPFILE=`mktemp`
trap 'unlink $PYTEMPFILE' EXIT INT QUIT ABRT

# yapf
yapf -version &> /dev/null || sudo pip install yapf 

# yapf
for dir in delta deltann dpl docker utils third_party;
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

while read file;
do
  echo "yapf: $file"
  yapf -i $file
done < $PYTEMPFILE


#clang-format
clang-format -version &> /dev/null || sudo apt-get install clang-format

CPPTEMPFILE=`mktemp`
trap 'unlink $CPPTEMPFILE' EXIT INT QUIT ABRT

# clang-format
for dir in delta deltann dpl docker utils third_party;
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


while read file;
do
  echo "clang-format: $file"
  clang-format -i $file 
done < $CPPTEMPFILE
