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

""" Create release notes with the issues from a milestone.
    python release_notes.py -c didi delta v.xxxxx
"""

import argparse
import urllib.request
import json
import collections

github_url = 'https://api.github.com/repos'

if __name__ == '__main__':

  # Parse arguments

  parser = argparse.ArgumentParser(
    description='Create a draft release with the issues from a milestone.'
  )

  parser.add_argument(
    'user',
    metavar='user',
    type=str,
    help='github user'
  )

  parser.add_argument(
    'repository',
    metavar='repository',
    type=str,
    help='githb repository'
  )

  parser.add_argument(
    'milestone',
    metavar='name of milestone',
    type=str,
    help='name of used milestone'
  )

  parser.add_argument(
    '-c', '--closed',
    help='Fetch closed milestones/issues',
    action='store_true'
  )

  args = parser.parse_args()

  # Fetch milestone id

  url = "%s/%s/%s/milestones" % (
    github_url,
    args.user,
    args.repository
  )

  headers = {
    'Origin': 'https://github.com',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '
                  'AppleWebKit/537.11 (KHTML, like Gecko) '
                  'Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive'}

  if args.closed:
    url += "?state=closed"
  req = urllib.request.Request(url, headers=headers)
  github_request = urllib.request.urlopen(req)

  if not github_request:
    parser.error('Cannot read milestone list.')

  decoder = json.JSONDecoder()

  milestones = decoder.decode(github_request.read().decode('utf-8'))

  print('parse milestones')

  github_request.close()

  milestone_id = None
  for milestone in milestones:
    if milestone['title'] == args.milestone:
      milestone_id = milestone['number']
  if not milestone_id:
    parser.error('Cannot find milestone')

  url = '%s/%s/%s/issues?milestone=%d' % (
    github_url,
    args.user,
    args.repository,
    milestone_id
  )

  if args.closed:
    url += "&state=closed"
  req = urllib.request.Request(url, headers=headers)
  github_request = urllib.request.urlopen(req)
  if not github_request:
    parser.error('Cannot read issue list.')

  issues = decoder.decode(github_request.read().decode('utf-8'))
  print('parse issues')
  github_request.close()

  final_data = []
  labels = []
  thanks_to = []
  for issue in issues:

    for label in issue['labels']:
      labels.append(label['name'])

    thanks_to.append('@%s' % (issue['user']['login']))
    final_data.append(' * **[%s]** - %s #%d by **@%s**\n' % (
      label['name'],
      issue['title'],
      issue['number'],
      issue['user']['login']
    ))

  dic = collections.defaultdict(set)
  for l_release in list(set(labels)):

    for f_data in final_data:
      if '[%s]' % l_release in f_data:
        dic[l_release].add(f_data)

  for key, value in dic.items():
    print('# %s\n%s' % (key, ''.join(value)))

  print('# %s\n%s' % ('Acknowledgements', 'Special thanks to %s ' % ('  '.join(list(set(thanks_to))))))
