#!/bin/bash

# Copyright 2016  Yajie Miao
# Apache 2.0

# Generate word-level alignment for a single utterance.

## Begin configuration section
stage=0
cmd=run.pl
num_threads=1

max_active=7000 # max-active
beam=15.0       # beam used
lattice_beam=8.0
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes

acoustic_scale=0.6  # the acoustic scales to be used
oov_word="<unk>"   # the oov word, used to convert oov words in the transcripts
# feature configurations; will be read from the training dir if not provided
norm_vars=
add_deltas=
## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Wrong #arguments ($#, expected 3)"
   echo "Usage: steps/align_ctc_single_utt.sh [options] <lang-dir> <data-dir> <uttdata-dir> <model-dir> <working-dir>"
   echo " e.g.: steps/align_ctc_single_utt.sh  data/lang_phn data/train data/uttdata exp/train_phn_l5_c320 exp/train_phn_l5_c320/align"
   echo "main options (for others, see top of script file)"
   echo "  --stage                                  # starts from which stage"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acoustic_scale                        # default 0.6 the value of acoustic scale to be used"
   exit 1;
fi

langdir=$1
data=$2
uttdata=$3
mdldir=$4
dir=`echo $5 | sed 's:/$::g'` # remove any trailing slash.

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

[ -z "$add_deltas" ] && add_deltas=`cat $mdldir/add_deltas 2>/dev/null`
[ -z "$norm_vars" ] && norm_vars=`cat $mdldir/norm_vars 2>/dev/null`

mkdir -p $dir/log

# Check if necessary files exist.
for f in $mdldir/final.nnet $mdldir/label.counts $data/feats.scp $uttdata/feats.scp $uttdata/text; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

## Set up the features
echo "$0: feature: norm_vars(${norm_vars}) add_deltas(${add_deltas})"
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:$uttdata/feats.scp ark:- |"
$add_deltas && feats="$feats add-deltas ark:- ark:- |"
##

## Create the "decoding" graph for this utterance
oov_int=`grep $oov_word $langdir/words.txt | awk '{print $2}'`

utils/sym2int.pl --map-oov $oov_int -f 2- $langdir/words.txt $uttdata/text > $dir/text_int 

utils/training_trans_fst.py $dir/text_int | fstcompile | fstarcsort --sort_type=olabel > $dir/G.fst

fsttablecompose ${langdir}/L.fst $dir/G.fst | fstdeterminizestar --use-log=true | \
  fstminimizeencoded | fstarcsort --sort_type=ilabel > $dir/LG.fst || exit 1;

fsttablecompose ${langdir}/T.fst $dir/LG.fst > $dir/TLG.fst || exit 1;

## Generate alignments
net-output-extract --class-frame-counts=$mdldir/label.counts --apply-log=true $mdldir/final.nnet "$feats" ark:- | \
  latgen-faster  --max-active=$max_active --max-mem=$max_mem --beam=$beam --lattice-beam=$lattice_beam \
  --acoustic-scale=$acoustic_scale --word-symbol-table=$langdir/words.txt --allow-partial=true $dir/TLG.fst ark:- ark:- | \
  lattice-1best --acoustic-scale=$acoustic_scale --ascale-factor=1 ark:- ark:- | \
  nbest-to-ctm ark:- - | \
  utils/int2sym.pl -f 5 $langdir/words.txt > $dir/ali

exit 0;
