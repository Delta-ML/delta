#!/usr/bin/env python3

# Apache 2.0

import sys

fread = open(sys.argv[1], 'r')

print('0 1 <eps> <eps>')
print('1 1 <blk> <eps>')
print('2 2 <blk> <eps>')
print('2 0 <eps> <eps>')

nodeX = 3
for entry in fread.readlines():
  entry = entry.replace('\n', '').strip()
  fields = entry.split(' ')
  phone = fields[0]
  if phone == '<eps>' or phone == '<blk>':
    continue

  if '#' in phone:
    print(str(0) + ' ' + str(0) + ' ' + '<eps>' + ' ' + phone)
  else:
    print(str(1) + ' ' + str(nodeX) + ' ' + phone + ' ' + phone)
    print(str(nodeX) + ' ' + str(nodeX) + ' ' + phone + ' <eps>')
    print(str(nodeX) + ' ' + str(2) + ' ' + '<eps> <eps>')
  nodeX += 1
print('0')

fread.close()
