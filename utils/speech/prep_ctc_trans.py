#!/usr/bin/env python3

# Copyright 2015       Yajie Miao    (Carnegie Mellon University)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This python script converts the word-based transcripts into label sequences. The labels are
# represented by their indices.

import sys

if __name__ == '__main__':

  if len(sys.argv) < 4 or len(sys.argv) > 5:
    print(
        "Usage: {0} <lexicon_file> <trans_file> <unk_word> [space_word]".format(
            sys.argv[0]))
    print(
        "e.g., utils/prep_ctc_trans.py data/lang/lexicon_numbers.txt data/train/text <UNK>"
    )
    print(
        "<lexicon_file> - the lexicon file in which entries have been represented by indices"
    )
    print("<trans_file>   - the word-based transcript file")
    print("<unk_word>     - the word which represents OOVs in transcripts")
    print(
        "[space_word]   - optional, the word representing spaces in the transcripts"
    )
    exit(1)

  dict_file = sys.argv[1]
  trans_file = sys.argv[2]
  unk_word = sys.argv[3]

  is_char = False
  if len(sys.argv) == 5:
    is_char = True
    space_word = sys.argv[4]

  # read the lexicon into a dictionary data structure
  fread = open(dict_file, 'r')
  dict = {}
  for line in fread.readlines():
    line = line.replace('\n', '')
    splits = line.split(' ')  # assume there are no multiple spaces
    word = splits[0]
    letters = ''
    for n in range(1, len(splits)):
      letters += splits[n] + ' '
    dict[word] = letters.strip()
  fread.close()

  # assume that each line is formatted as "uttid word1 word2 word3 ...", with no multiple spaces appearing
  fread = open(trans_file, 'r')
  for line in fread.readlines():
    out_line = ''
    line = line.replace('\n', '').strip()
    while '  ' in line:
      line = line.replace('  ',
                          ' ')  # remove multiple spaces in the transcripts

    uttid = line.split(' ')[0]  # the first field is always utterance id
    trans = line.replace(uttid, '').strip()
    if is_char:
      trans = trans.replace(' ', ' ' + space_word + ' ')
    splits = trans.split(' ')

    out_line += uttid + ' '
    for n in range(0, len(splits)):
      try:
        out_line += dict[splits[n]] + ' '
      except Exception:
        out_line += dict[unk_word] + ' '
    print(out_line.strip())
