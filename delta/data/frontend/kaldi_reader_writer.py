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

import kaldiio

class BaseWriter:
  def __setitem__(self, key, value):
    raise NotImplementedError

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def close(self):
    try:
      self.writer.close()
    except Exception:
      pass

    if self.writer_scp is not None:
      try:
        self.writer_scp.close()
      except Exception:
        pass

      if self.writer_nframe is not None:
        try:
          self.writer_nframe.close()
        except Exception:
          pass

def get_num_frames_writer(write_num_frames: str):
  """get_num_frames_writer

  Examples:
      >>> get_num_frames_writer('ark,t:num_frames.txt')
  """
  if write_num_frames is not None:
    if ':' not in write_num_frames:
      raise ValueError('Must include ":", write_num_frames={}'
                       .format(write_num_frames))

    nframes_type, nframes_file = write_num_frames.split(':', 1)
    if nframes_type != 'ark,t':
      raise ValueError(
        'Only supporting text mode. '
        'e.g. --write-num-frames=ark,t:foo.txt :'
        '{}'.format(nframes_type))

  return open(nframes_file, 'w', encoding='utf-8')

class KaldiWriter(BaseWriter):
  def __init__(self, wspecifier, write_num_frames=None, compress=False,
               compression_method=2):
    if compress:
      self.writer = kaldiio.WriteHelper(
        wspecifier, compression_method=compression_method)
    else:
      self.writer = kaldiio.WriteHelper(wspecifier)
    self.writer_scp = None
    if write_num_frames is not None:
      self.writer_nframe = get_num_frames_writer(write_num_frames)
    else:
      self.writer_nframe = None

  def __setitem__(self, key, value):
    self.writer[key] = value
    if self.writer_nframe is not None:
      self.writer_nframe.write(f'{key} {len(value)}\n')
