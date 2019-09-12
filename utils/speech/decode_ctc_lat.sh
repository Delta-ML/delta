#!/bin/bash

# Apache 2.0

# Decode the CTC-trained model by generating lattices.


## Begin configuration section
stage=0
nj=16
cmd=run.pl
num_threads=1

acwt=0.9
min_active=200
max_active=7000 # max-active
beam=15.0       # beam used
lattice_beam=8.0
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes
mdl=final.nnet
blank_scale=1.0
label_counts=
block_softmax=

skip_scoring=false # whether to skip WER scoring
scoring_opts="--min-acwt 5 --max-acwt 10 --acwt-factor 0.1"
score_with_conf=false

# feature configurations; will be read from the training dir if not provided
norm_vars=
add_deltas=
subsample_feats=
splice_feats=
subsample_frames=2
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
   echo "  --acwt                                   # default 0.9, the acoustic scale to be used"
   exit 1;
fi

graphdir=$1
data=$2
dir=`echo $3 | sed 's:/$::g'` # remove any trailing slash.

srcdir=`dirname $dir`; # assume model directory one level up from decoding directory.
sdata=$data/split$nj;

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"
[ -z "$label_counts" ] && label_counts=${srcdir}/label.counts
[ -z "$block_softmax" ] && bs=""
[ -z "$block_softmax" ] || bs="--blockid=${block_softmax}"

[ -z "$add_deltas" ] && add_deltas=`cat $srcdir/add_deltas 2>/dev/null`
[ -z "$norm_vars" ] && norm_vars=`cat $srcdir/norm_vars 2>/dev/null`
[ -z "$subsample_feats" ] && subsample_feats=`cat $srcdir/subsample_feats 2>/dev/null` || subsample_feats=false
[ -z "$splice_feats" ] && splice_feats=`cat $srcdir/splice_feats 2>/dev/null` || splice_feats=false

mkdir -p $dir/log
split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Check if necessary files exist.
for f in $graphdir/TLG.fst $label_counts $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
if [ -f $mdl ] && [ `strings $mdl | grep -c "<BiLstmParallel>"` -gt 0 ]; then
    tmpdir=`mktemp -d`
    echo model $mdl appears parallel, converting in $tmpdir
    format-to-nonparallel $mdl $tmpdir/model.nnet
    mdl=$tmpdir/model.nnet
    trap "rm -rf $tmpdir" EXIT
else
    mdl=$srcdir/$mdl
fi

## Set up the features
echo "$0: feature: norm_vars(${norm_vars}) add_deltas(${add_deltas}) subsample_feats(${subsample_feats}) splice_feats(${splice_feats})"
feats="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
$splice_feats && feats="$feats splice-feats --left-context=1 --right-context=1 ark:- ark:- |"
$subsample_feats && feats="$feats subsample-feats --n=$subsample_frames --offset=0 ark:- ark:- |"
$add_deltas && feats="$feats add-deltas ark:- ark:- |"
##

# Decode for each of the acoustic scales
$cmd JOB=1:$nj $dir/log/decode.JOB.log \
  net-output-extract --class-frame-counts=$label_counts --apply-log=true $bs --blank-scale=$blank_scale $mdl "$feats" ark:- \| tee Aeval.JOB.ark \| \
  latgen-faster  --max-active=$max_active --max-mem=$max_mem --beam=$beam --lattice-beam=$lattice_beam \
  --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $graphdir/TLG.fst ark:- "ark:|gzip -c > $dir/lat.JOB.gz" || \
exit 1;

# Scoring
if ! $skip_scoring ; then
  if [ -f $data/stm ]; then # use sclite scoring.
    if $score_with_conf ; then
      [ ! -x local/score_sclite_conf.sh ] && echo "Not scoring because local/score_sclite_conf.sh does not exist or not executable." && exit 1;
      local/score_sclite_conf.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
    else
      [ ! -x local/score_sclite.sh ] && echo "Not scoring because local/score_sclite.sh does not exist or not executable." && exit 1;
      local/score_sclite.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
    fi
  else
    [ ! -x local/score.sh ] && echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir || exit 1;
  fi
fi

exit 0;
