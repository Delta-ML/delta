
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
