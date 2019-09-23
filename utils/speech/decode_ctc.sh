#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#           2015  Yajie Miao    (Carnegie Mellon University)
# Apache 2.0

# Decode the CTC-trained model. Currently we are using the simplest best-path decoding. Lattice-based 
# decoding and other formats of decoding outputs (e.g., CTM) will be added in our future development.  


## Begin configuration section
stage=0
nj=16
cmd=run.pl
num_threads=1

max_active=7000 # max-active
beam=15.0 # beam used

skip_scoring=false # whether to skip WER scoring
acoustic_scales="0.5 0.6 0.7 0.8"  # the acoustic scales to be used

# feature configurations; will be read from the training dir if not provided
norm_vars=
add_deltas=
## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: steps/decode_ctc.sh [options] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_ctc.sh data/lang data/test exp/train_l4_c320/decode"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acoustic_scales                        # default 0.5 0.6 0.7 0.8 ... the values of acoustic scales to be used"
   exit 1;
fi

graphdir=$1
data=$2
dir=`echo $3 | sed 's:/$::g'` # remove any trailing slash.

srcdir=`dirname $dir`; # assume model directory one level up from decoding directory.
sdata=$data/split$nj;

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

[ -z "$add_deltas" ] && add_deltas=`cat $srcdir/add_deltas 2>/dev/null`
[ -z "$norm_vars" ] && norm_vars=`cat $srcdir/norm_vars 2>/dev/null`

mkdir -p $dir/log
split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Check if necessary files exist.
for f in $graphdir/TLG.fst $srcdir/label.counts $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

## Set up the features
echo "$0: feature: norm_vars(${norm_vars}) add_deltas(${add_deltas})"
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
$add_deltas && feats="$feats add-deltas ark:- ark:- |"
##

# Decode for each of the acoustic scales
for ascale in $acoustic_scales; do
  echo "$0: decoding with acoustic scale $ascale"
  $cmd JOB=1:$nj $dir/log/decode.$ascale.JOB.log \
    net-output-extract --class-frame-counts=$srcdir/label.counts --apply-log=true $srcdir/final.nnet "$feats" ark:- \| \
    decode-faster --beam=$beam --max-active=$max_active --acoustic-scale=$ascale --word-symbol-table=$graphdir/words.txt \
    --allow-partial=true $graphdir/TLG.fst ark:- ark,t:$dir/trans.$ascale.JOB 
  cat $dir/trans.$ascale.* > $dir/trans.$ascale
  rm -f $dir/trans.$ascale.*
done

# Scoring
cat $data/text | sed 's:<UNK>::g' | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' > $dir/text_filt 
if ! $skip_scoring ; then
  for ascale in $acoustic_scales; do
     cat $dir/trans.$ascale | utils/int2sym.pl -f 2- $graphdir/words.txt | \
       sed 's:<UNK>::g' | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' | \
       compute-wer --text --mode=present ark:$dir/text_filt  ark,p:-  >& $dir/wer_$ascale || exit 1;
  done
fi

exit 0;
