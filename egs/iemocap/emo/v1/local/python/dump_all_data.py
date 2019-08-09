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

''' dump impro and scprit to `all` data'''

import os
import sys

def gen_all_data(all_path, path):
  for root, dirs, files in os.walk(path):
    for file in files:
      filepath = os.path.join(root, file)
      l = filepath.split('/')
      if 'impro' in l:
          index = l.index('impro') 
      elif 'script' in l:
          index = l.index('script')
      else:
        print('Error,', path)    
        exit(-1)

      l = l[index + 1:]
      link_file = os.path.join(all_path, '/'.join(l))
      link_path = os.path.dirname(link_file)
  
      filepath = os.path.relpath(filepath, link_path)
  
      if not os.path.exists(link_path):
          os.makedirs(link_path)
      if os.path.exists(link_file):
          os.remove(link_file)
  
      os.symlink(filepath, link_file)



if __name__ == '__main__':
  if 4 != len(sys.argv):
    print('Usage : ', sys.argv[0], ' impro_path script_path all_path')
    exit(-1)
  
  impro_path = sys.argv[1]
  script_path = sys.argv[2]
  all_path = sys.argv[3]
  
  gen_all_data(all_path, impro_path)
  gen_all_data(all_path, script_path)
