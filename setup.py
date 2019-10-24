from setuptools import setup, find_packages, Extension
from datetime import date
import os
import sys
from glob import glob
import tensorflow as tf
from absl import logging

logging.set_verbosity(logging.info)


TF_INCLUDE, TF_CFLAG = tf.sysconfig.get_compile_flags()
TF_INCLUDE = TF_INCLUDE.split('-I')[1]

TF_LIB_INC, TF_SO_LIB = tf.sysconfig.get_link_flags()
TF_SO_LIB = TF_SO_LIB.replace('-l:libtensorflow_framework.1.dylib',
                              '-ltensorflow_framework.1')
TF_LIB_INC = TF_LIB_INC.split('-L')[1]
TF_SO_LIB = TF_SO_LIB.split('-l')[1]

logging.info("TF_INCLUDE: {}".format(TF_INCLUDE))
logging.info("TF_CFLAG: {}".format(TF_CFLAG))
logging.info("TF_LIB_INC: {}".format(TF_LIB_INC))
logging.info("TF_SO_LIB: {}".format(TF_SO_LIB))

NAME = "delta-didi"
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
PLATFORMS = ["MacOS", "Unix"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3.6",
]
this_directory = os.path.abspath(os.path.dirname(__file__))


def get_long_description():
  try:
    with open(os.path.join(this_directory, 'README.md'),
              encoding='utf-8') as f:
      long_description = f.read()
  except:
    long_description = "No long description!"
  return long_description


def get_license():
  try:
    with open(os.path.join(this_directory, 'LICENSE'),
              encoding='utf-8') as f:
      license = f.read()
  except:
    logging.info("license not found in '%s.__init__.py'!" % NAME)
    license = ""
  return license


def get_requires():
  try:
    f = open("requirements.txt")
    requires = [i.strip() for i in f.read().split("\n")]
  except:
    logging.info("'requirements.txt' not found!")
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
long_description = get_long_description()
license_ = get_license()
packages = find_packages()
logging.info("long_description: {}".format(long_description))
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
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.2",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    packages=packages,
    package_data={"delta": ["resources/cppjieba_dict/*.utf8"]},
    entry_points={
        'console_scripts': [
            'delta = delta.main:entry']},
    url=URL,
    download_url=DOWNLOAD_URL,
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    license=license_,
    install_requires=get_requires(),
    ext_modules=[module]
)
