#!/usr/bin/env python

# Init from espnet: https://github.com/espnet/espnet
# modify to support tensorflow
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import importlib
import logging
import sys

# you should add the libraries which are not included in setup.py
MANUALLY_INSTALLED_LIBRARIES = [
    ('kaldiio', None),
    ('matplotlib', None),
    ('librosa', None),
    ('sklearn', None),
    ('pandas', None),
    ('soundfile', None),
    ('textgrid', None),
    ('yapf', None),
    ('jieba', None),
    ('yaml', None),
    ('absl', None),
    ('hurry.filesize', None),
    ('gensim', None),
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

logging.info("python version = " + sys.version)

library_list = []
library_list.extend(MANUALLY_INSTALLED_LIBRARIES)

# check library availableness
logging.info("library availableness check start.")
logging.info("# libraries to be checked = %d" % len(library_list))
is_correct_installed_list = []
for idx, (name, version) in enumerate(library_list):
  try:
    importlib.import_module(name)
    logging.info("--> %s is installed." % name)
    is_correct_installed_list.append(True)
  except ImportError:
    logging.warning("--> %s is not installed." % name)
    is_correct_installed_list.append(False)
logging.info("library availableness check done.")
logging.info("%d / %d libraries are correctly installed." %
             (sum(is_correct_installed_list), len(library_list)))

if len(library_list) != sum(is_correct_installed_list):
  logging.info("please try to setup again and then re-run this script.")
  sys.exit(1)

# check library version
num_version_specified = sum(
    [True if v is not None else False for n, v in library_list])
logging.info("library version check start.")
logging.info("# libraries to be checked = %d" % num_version_specified)
is_correct_version_list = []
for idx, (name, version) in enumerate(library_list):
  if version is not None:
    lib = importlib.import_module(name)
    if hasattr(lib, "__version__"):
      is_correct = lib.__version__ in version
      if is_correct:
        logging.info("--> %s version is matched." % name)
        is_correct_version_list.append(True)
      else:
        logging.warning("--> %s version is not matched (%s is not in %s)." %
                        (name, lib.__version__, str(version)))
        is_correct_version_list.append(False)
    else:
      logging.info("--> %s has no version info, but version is specified." %
                   name)
      logging.info("--> maybe it is better to reinstall the latest version.")
      is_correct_version_list.append(False)
logging.info("library version check done.")
logging.info("%d / %d libraries are correct version." %
             (sum(is_correct_version_list), num_version_specified))

if sum(is_correct_version_list) != num_version_specified:
  logging.info("please try to setup again and then re-run this script.")
  sys.exit(1)

# check cuda availableness
logging.info("cuda availableness check start.")
import delta.compat as tf
try:
  assert tf.test.is_gpu_available()
  logging.info("--> cuda is available in tensorflow.")
except AssertionError:
  logging.warning("--> it seems that cuda is not available in tensorflow.")

try:
  assert len(tf.test.gpu_device_name()) > 1
  logging.info("--> multi-gpu is available (#gpus = %d)." %
               len(tf.test.gpu_device_name()))
except AssertionError:
  logging.warning("--> it seems that only single gpu is available.")
  logging.warning('--> maybe your machine has only one gpu.')
logging.info("cuda availableness check done.")
