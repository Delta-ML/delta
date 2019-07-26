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

#pylint: disable=bad-indentation
#!/usr/bin/env python3

import os
import sys
import codecs
import numpy as np
import textgrid

SAMPLE_FREQ = 16000
SAMPLE_DTYPE = 'int16'
SAMPLE_DBYTE = 2

if __name__ == '__main__':
  if len(sys.argv) < 5:
    print('Usage: %s pcm_file textgrid_file out_trans_file out_pcm_dir')
    exit(1)

  pcm_file = sys.argv[1]
  textgrid_file = sys.argv[2]
  out_trans_file = sys.argv[3]
  out_pcm_dir = sys.argv[4]

  print('Loading PCM file: %s' % (pcm_file))
  pcm_file_basename = os.path.split(pcm_file)[-1]
  pcm_file_basename_no_postfix = pcm_file_basename.split('.')[0]
  print('PCM file name (no postfix): ', pcm_file_basename_no_postfix)

  with open(pcm_file, 'rb') as fp_in:
    pcm_array = np.frombuffer(fp_in.read(), dtype=SAMPLE_DTYPE)
  pcm_length_sec = pcm_array.shape[0] / SAMPLE_FREQ
  print('PCM length: %d:%d' % (pcm_length_sec / 60, pcm_length_sec % 60))

  print('Loading TextGrid file and output PCM/trans: %s' % (textgrid_file))
  with codecs.open(textgrid_file, 'r', encoding='utf-16') as fp_in:
    with codecs.open(out_trans_file, 'w', encoding='gb18030') as fp_out_trans:
      the_grid = textgrid.TextGrid(fp_in.read())
      for idx, tier in enumerate(the_grid):
        print(idx)
        print(tier.size)
        print(tier.xmin)
        print(tier.xmax)
        print(tier.nameid)

        seg_idx = 0
        for xmin, xmax, text in tier.simple_transcript:
          print(xmin, xmax, end=' ')
          print(codecs.encode(text, 'utf-8'))

          xmin = float(xmin)
          xmax = float(xmax)

          seg_str_esc = '%.2f-%.2f' % (xmin, xmax)

          # index into PCM array and output .pcm file for this segment
          xmin_samples = int(xmin * SAMPLE_FREQ)
          xmax_samples = int(xmax * SAMPLE_FREQ)
          seg_array = pcm_array[xmin_samples:xmax_samples]

          output_file_name = '%s__%s__%s.pcm' % (pcm_file_basename, seg_idx,
                                                 seg_str_esc)
          output_file_path = os.path.join(out_pcm_dir, output_file_name)
          print(output_file_path)
          fp_out_trans.write(output_file_path)
          fp_out_trans.write(':')
          fp_out_trans.write(text)
          fp_out_trans.write('\n')
          with open(output_file_path, 'wb') as fp_out:
            fp_out.write(np.ascontiguousarray(seg_array))

          seg_idx += 1
