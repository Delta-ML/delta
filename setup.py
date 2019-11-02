from setuptools import setup
from setuptools import find_packages
from setuptools import Extension
from datetime import date
import os
import sys
from glob import glob
import tensorflow as tf
from absl import logging

logging.set_verbosity(logging.INFO)


TF_INCLUDE, TF_CFLAG = tf.sysconfig.get_compile_flags()
TF_INCLUDE = TF_INCLUDE.split('-I')[1]

TF_LIB_INC, TF_SO_LIB = tf.sysconfig.get_link_flags()
TF_SO_LIB = TF_SO_LIB.replace('-l:libtensorflow_framework.2.dylib',
                              '-ltensorflow_framework.2')
TF_LIB_INC = TF_LIB_INC.split('-L')[1]
TF_SO_LIB = TF_SO_LIB.split('-l')[1]

logging.info("TF_INCLUDE: {}".format(TF_INCLUDE))
logging.info("TF_CFLAG: {}".format(TF_CFLAG))
logging.info("TF_LIB_INC: {}".format(TF_LIB_INC))
logging.info("TF_SO_LIB: {}".format(TF_SO_LIB))

NAME = "delta-nlp"
GITHUB_USER_NAME = "didi"
AUTHOR = "Speech@DiDi"
AUTHOR_EMAIL = "speech@didiglobal.com"
MAINTAINER = "applenob"
MAINTAINER_EMAIL = "chenjunwen@didiglobal.com"
REPO_NAME = os.path.basename(os.getcwd())
URL = "https://github.com/{0}/{1}".format(GITHUB_USER_NAME, REPO_NAME)
GITHUB_RELEASE_TAG = str(date.today())
DOWNLOAD_URL = "https://github.com/{0}/{1}/tarball/{2}".format(
    GITHUB_USER_NAME, REPO_NAME, GITHUB_RELEASE_TAG)
SHORT_DESCRIPTION = "DELTA is a deep learning based natural language and speech processing platform."
LONG_DESCRIPTION = """
# DELTA - A DEep learning Language Technology plAtform

DELTA is a deep learning based end-to-end natural language and speech processing platform. DELTA aims to provide easy and fast experiences for using, deploying, and developing natural language processing and speech models for both academia and industry use cases. DELTA is mainly implemented using TensorFlow and Python 3.

Refer to github for more information: https://github.com/didi/delta
"""
PLATFORMS = ["MacOS", "Unix"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3.6"
]
this_directory = os.path.abspath(os.path.dirname(__file__))


def get_requires():
  require_file = "tools/requirements.txt"
  try:
    f = open(require_file)
    requires = [i.strip() for i in f.read().split("\n")]
  except:
    logging.info("{} not found!".format(require_file))
    requires = list()
  return requires


complie_args = [TF_CFLAG, "-fPIC", "-shared", "-O2", "-std=c++11"]
if sys.platform == 'darwin':  # Mac os X before Mavericks (10.9)
  complie_args.append("-stdlib=libc++")
cppjieba_includes = ["tools/cppjieba/deps",
                     "tools/cppjieba/include"]
include_dirs = ['delta', 'delta/layers/ops/', TF_INCLUDE] + cppjieba_includes

module = Extension('delta.layers.ops.x_ops',
                   sources=glob('delta/layers/ops/kernels/*.cc'),
                   extra_compile_args=complie_args,
                   include_dirs=include_dirs,
                   library_dirs=[TF_LIB_INC],
                   libraries=[TF_SO_LIB],
                   language='c++')
license_ = "Apache Software License"
packages = find_packages()
logging.info("LONG_DESCRIPTION: {}".format(LONG_DESCRIPTION))
logging.info("license: {}".format(license_))
logging.info("packages: {}".format(packages))

custom_op_files = glob("delta/layers/ops/x_ops*.so")
if len(custom_op_files) > 0:
  for custom_op_file in custom_op_files:
    if os.path.exists(custom_op_file):
      logging.info("Remove file {}.".format(custom_op_file))
      os.remove(custom_op_file)

setup(
    name=NAME,
    description=SHORT_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    version="0.2.1",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    packages=packages,
    package_data={"delta": ["resources/cppjieba_dict/*.utf8"]},
    entry_points={
        'console_scripts': [
            'delta = delta.main:nlp_entry']},
    url=URL,
    download_url=DOWNLOAD_URL,
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    license=license_,
    install_requires=get_requires(),
    ext_modules=[module]
)
