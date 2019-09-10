#!/bin/bash
# prepare dictionary for HKUST
# it is done for English and Chinese separately, 
# For English, we use CMU dictionary, and Sequitur G2P
# for OOVs, while all englist phone set will concert to Chinese
# phone set at the end. For Chinese, we use an online dictionary,
# for OOV, we just produce pronunciation using Charactrt Mapping.
  
. path.sh

[ $# != 0 ] && echo "Usage: local/hkust_prepare_dict.sh" && exit 1;

train_dir=data/local/train
dev_dir=data/local/dev
dict_dir=data/local/dict
mkdir -p $dict_dir

case 0 in    #goto here
    1)
;;           #here:
esac


# extract full vocabulary
cat $train_dir/text $dev_dir/text | awk '{for (i = 2; i <= NF; i++) print $i}' |\
  sed -e 's/ /\n/g' | sort -u | grep -v '\[LAUGHTER\]' | grep -v '\[NOISE\]' |\
  grep -v '\[VOCALIZED-NOISE\]' > $dict_dir/vocab-full.txt  

# split into English and Chinese
cat $dict_dir/vocab-full.txt | grep '[a-zA-Z]' > $dict_dir/vocab-en.txt
cat $dict_dir/vocab-full.txt | grep -v '[a-zA-Z]' > $dict_dir/vocab-ch.txt

# produce pronunciations for english 
if [ ! -f $dict_dir/cmudict/cmudict.0.7a ]; then
  echo "--- Downloading CMU dictionary ..."
  svn co https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict \
    $dict_dir/cmudict || exit 1;
fi

echo "--- Striping stress and pronunciation variant markers from cmudict ..."
perl $dict_dir/cmudict/scripts/make_baseform.pl \
  $dict_dir/cmudict/cmudict.0.7a /dev/stdout |\
  sed -e 's:^\([^\s(]\+\)([0-9]\+)\(\s\+\)\(.*\):\1\2\3:' > $dict_dir/cmudict-plain.txt

echo "--- Searching for English OOV words ..."
gawk 'NR==FNR{words[$1]; next;} !($1 in words)' \
  $dict_dir/cmudict-plain.txt $dict_dir/vocab-en.txt |\
  egrep -v '<.?s>' > $dict_dir/vocab-en-oov.txt

gawk 'NR==FNR{words[$1]; next;} ($1 in words)' \
  $dict_dir/vocab-en.txt $dict_dir/cmudict-plain.txt |\
  egrep -v '<.?s>' > $dict_dir/lexicon-en-iv.txt

wc -l $dict_dir/vocab-en-oov.txt
wc -l $dict_dir/lexic