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

#!/usr/bin/env python3
#pylint: disable=bad-indentation

import os
import sys
import codecs
import textgrid

SAMPLE_FREQ = 8000
SAMPLE_DTYPE = 'int16'
SAMPLE_DBYTE = 2

TEXTGRID_SUFFIX = 'textgrid'
PCM_SUFFIX = 'wav'
VALID_SEGMENT_TEXTGRID_LABEL = '#'
VALID_SEGMENT_OUTPUT_LABEL = '1'

if __name__ == '__main__':
  if len(sys.argv) < 3:
    print('Usage: %s textgrid_file_list output_segment_list')
    exit(1)

  textgrid_file_list = sys.argv[1]
  output_segment_list = sys.argv[2]

  with open(textgrid_file_list) as fp_in, open(output_segment_list,
                                               'w') as fp_out:
    for line in fp_in:
      textgrid_file = line.strip()
      if textgrid_file[-len(TEXTGRID_SUFFIX):].lower() != TEXTGRID_SUFFIX:
        print('File %s does not have textgrid suffix, skipped.' %
              (textgrid_file))
        continue

      base_file = textgrid_file[:-len(TEXTGRID_SUFFIX)]
      pcm_file = base_file + PCM_SUFFIX
      '''
            # unfinished code to check PCM length
            print('Loading PCM file: %s' % (pcm_file))
            pcm_file_basename = os.path.split(pcm_file)[-1]
            pcm_file_basename_no_postfix = pcm_file_basename.split('.')[0]
            print('PCM file name (no postfix): ', pcm_file_basename_no_postfix)

            with open(pcm_file, 'rb') as fp_in:
                pcm_array = np.frombuffer(fp_in.read(), dtype = SAMPLE_DTYPE)
            pcm_length_sec = pcm_array.shape[0] / SAMPLE_FREQ
            print('PCM length: %d:%d' % (pcm_length_sec / 60, pcm_length_sec % 60))
            '''
      print('Loading TextGrid file: %s' % (textgrid_file))
      with codecs.open(textgrid_file, 'r', encoding='utf-8') as fp_in:
        fp_out.write('%s' % (os.path.abspath(pcm_file)))
        the_grid = textgrid.TextGrid(fp_in.read())
        for idx, tier in enumerate(the_grid):
          #print(idx)
          #print(tier.size)
          #print(tier.xmin)
          #print(tier.xmax)
          #print(tier.nameid)

          seg_idx = 0
          for xmin, xmax, text in tier.simple_transcript:
            #print(xmin, xmax, end=' ')
            #print(codecs.encode(text, 'utf-8'))

            xmin = float(xmin)
            xmax = float(xmax)

            if text == VALID_SEGMENT_TEXTGRID_LABEL:
              fp_out.write(' (%f,%f)' % (xmin, xmax))

            seg_idx += 1

        fp_out.write('\n')

print('Done.')
