#!/bin/bash

apt-get update && apt-get install -y \
  clang-format \
  curl \
  git \
  sudo \
  sox \
  tig \
  vim 


# fix tf1.14 docker image for issue
# https://github.com/tensorflow/tensorflow/issues/29951
apt-get install -y gcc-4.8 g++-4.8


