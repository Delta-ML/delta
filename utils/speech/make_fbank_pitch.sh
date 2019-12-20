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

#default params
nj=1
cmd=utils/run.pl
sample_rate=16000
upper_frequency_limit=4000
lower_frequency_limit=20
filterbank_channel_count=40
window_length=0.025
frame_length=0.010
output_type=1
snip_edges=true
raw_energy=1
preeph_coeff=0.97
window_type='povey'
remove_dc_offset=true
is_fbank=true
dither=0.0
thres_autoc=0.3
write_utt2num_frames=true
compress=false
compression_method=2

if [ -f path.sh ]; then . ./path.sh; fi
 . parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
  cat >&2 <<EOF
Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
 e.g.: $0 data/train
Note: <log-dir> defaults to <data-dir>/log, and
      <fbank-dir> defaults to <data-dir>/data
Options:
  --nj <nj>                            # number of parallel jobs.
  --cmd <run.pl|queue.pl <queue opts>> # how to run jobs.
  --write_utt2num_frames <true|false>  # If true, write utt2num_frames file.
EOF
   exit 1;
fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  fbank_pitch_dir=$3
else
  fbank_pitch_dir=$data/data
fi

# make $fbank_pitch_dir an absolute pathname.
fbank_pitch_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $fbank_pitch_dir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $fbank_pitch_dir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

utils/validate_data_dir.sh --no-text --no-feats ${data} || exit 1;

split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav.${n}.scp"
done

utils/split_scp.pl ${scp} ${split_scps} || exit 1;

if ${write_utt2num_frames}; then
  write_num_frames_opt="--write_num_frames=ark,t:${logdir}/utt2num_frames.JOB"
else
  write_num_frames_opt=
fi

ext=ark

if [ -f ${data}/segments ]; then
    echo "$0 [info]: segments file exists: using that."
    split_segments=""
    for n in $(seq ${nj}); do
        split_segments="${split_segments} ${logdir}/segments.${n}"
    done

    utils/split_scp.pl ${data}/segments ${split_segments}

    ${cmd} JOB=1:${nj} ${logdir}/make_fbank_pitch${name}.JOB.log \
        speech/compute_fbank_pitch.py \
            --sample_rate ${sample_rate} \
            --upper_frequency_limit ${upper_frequency_limit} \
            --lower_frequency_limit ${lower_frequency_limit} \
            --filterbank_channel_count ${filterbank_channel_count} \
            --window_length ${window_length} \
            --frame_length ${frame_length} \
            --thres_autoc ${thres_autoc} \
            --output_type ${output_type} \
            --snip_edges ${snip_edges} \
            --raw_energy ${raw_energy} \
            --preeph_coeff ${preeph_coeff} \
            --window_type ${window_type} \
            --remove_dc_offset ${remove_dc_offset} \
            --is_fbank ${is_fbank} \
            --dither ${dither} \
            ${write_num_frames_opt} \
            --compress ${compress} \
            --compression_method ${compression_method} \
            --segment=${logdir}/segments.JOB scp:${scp} \
            ark,scp:${fbank_pitch_dir}/raw_fbank_pitch${name}.JOB.${ext},${fbank_pitch_dir}/raw_fbank_pitch${name}.JOB.scp || exit 1

else
  echo "$0: [info]: no segments file exists: assuming pcm.scp indexed by utterance."
  split_scps=""
  for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav.${n}.scp"
  done

  utils/split_scp.pl ${scp} ${split_scps}

  ${cmd} JOB=1:${nj} ${logdir}/make_fbank_pitch${name}.JOB.log \
      speech/compute_fbank_pitch.py \
            --sample_rate ${sample_rate} \
            --upper_frequency_limit ${upper_frequency_limit} \
            --lower_frequency_limit ${lower_frequency_limit} \
            --filterbank_channel_count ${filterbank_channel_count} \
            --window_length ${window_length} \
            --frame_length ${frame_length} \
            --thres_autoc ${thres_autoc} \
            --output_type ${output_type} \
            --snip_edges ${snip_edges} \
            --raw_energy ${raw_energy} \
            --preeph_coeff ${preeph_coeff} \
            --window_type ${window_type} \
            --remove_dc_offset ${remove_dc_offset} \
            --is_fbank ${is_fbank} \
            --dither ${dither} \
            ${write_num_frames_opt} \
            --compress ${compress} \
            --compression_method ${compression_method} \
            scp:${logdir}/wav.JOB.scp \
            ark,scp:${fbank_pitch_dir}/raw_fbank_pitch${name}.JOB.${ext},${fbank_pitch_dir}/raw_fbank_pitch${name}.JOB.scp || exit 1
fi

# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat ${fbank_pitch_dir}/raw_fbank_pitch${name}.${n}.scp || exit 1;
done > ${data}/feats.scp || exit 1

if ${write_utt2num_frames}; then
    for n in $(seq ${nj}); do
        cat ${logdir}/utt2num_frames.${n} || exit 1;
    done > ${data}/utt2num_frames || exit 1
    rm ${logdir}/utt2num_frames.* 2>/dev/null
fi

rm -f ${logdir}/wav.*.scp ${logdir}/segments.* 2>/dev/null

# Write the filetype, this will be used for data2json.sh
echo ${filetype} > ${data}/filetype

nf=$(wc -l < ${data}/feats.scp)
nu=$(wc -l < ${data}/wav.scp)
if [ ${nf} -ne ${nu} ]; then
    echo "It seems not all of the feature files were successfully ($nf != $nu);"
    echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating filterbank_picth features for $name"
