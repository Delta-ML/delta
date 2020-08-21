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

SHELL = /bin/bash
PATH  := ~/.local/bin:$(PATH)

# Use a prebuild Kaldi to omit the installation
KALDI :=

# Both Miniconda2/3 can install any Python versions
GCC_VERSION := $(shell gcc -dumpversion)

ifeq ($(shell id -u),0)
	SUDO :=
else
	SUDO := sudo
endif

.PHONY: all clean

all: delta deltann

# delta
delta: basic speech check_install

basic: linux.done python.done tensorflow.done espnet.done nlp_zh compile_ops.done

speech: kaldi.done

nlp: python_nlp.done nlp_zh compile_ops.done

nlp_zh: cppjieba.done

extra: sentencepiece.done moses.done

PIP_INSTALL=python3 -m pip --no-cache-dir install

APT_INSTALL=apt-get install -y --no-install-recommends

PYTHON=python3

# test
PROTOC_ZIP := protoc-3.7.1-linux-x86_64.zip
protoc:
	$(SUDO) apt-get update && $(SUDO) $(APT_INSTALL) curl unzip
	curl -OL --retry 100 https://github.com/google/protobuf/releases/download/v3.7.1/$(PROTOC_ZIP) &&\
	$(SUDO) unzip -o $(PROTOC_ZIP) -d /usr/local bin/protoc &&\
	rm -f $(PROTOC_ZIP)

# only for test cpp
test.deps:
	test -d abseil-cpp || git clone --depth=1 https://github.com/abseil/abseil-cpp.git
	test -d googletest || git clone --depth=1 https://github.com/google/googletest.git 

test: test.deps #protoc
	make -C test

# install python packages
python.done:
	$(PIP_INSTALL) --no-cache-dir --user -r requirements.txt
	if [ -f /etc/lsb-release ]; then $(SUDO) $(APT_INSTALL) libsndfile1; elif [[ "$$OSTYPE" == "darwin"* ]]; then brew install libsndfile; else $(SUDO) yum install -y libsndfile-devel; fi;
	touch python.done

python_nlp.done:
	$(PIP_INSTALL) --user -r requirements.txt
	touch python_nlp.done

linux.done:
	if [ -f /etc/lsb-release ]; then $(SUDO) apt-get update && $(SUDO) $(APT_INSTALL) bc sox; elif [[ "$$OSTYPE" == "darwin"* ]]; then brew install bc sox; else $(SUDO) yum install -y bc sox; fi;
	touch linux.done

# compile custom ops on your environment
compile_ops.done:
	/bin/bash -c 'pushd ../transform/tf_wrapper/ops; bash build.sh delta'
	touch compile_ops.done

# only for speech
ifneq ($(strip $(KALDI)),)
kaldi.done:
	ln -s $(abspath $(KALDI)) kaldi
	touch kaldi.done
else
kaldi.done:
	test -d kaldi || git clone --depth=1 https://github.com/kaldi-asr/kaldi.git
	#cd kaldi/tools; ./extras/install_mkl.sh; $(MAKE) -j $(nproc)
	#cd kaldi/src; ./configure --use-cuda=no; $(MAKE) depend -j $(nproc); $(MAKE) -j $(nproc)
	cd kaldi/tools; ./extras/install_openblas.sh; $(MAKE) -j $(nproc)
	cd kaldi/src; ./configure --static --mathlib=OPENBLAS --use-cuda=no; $(MAKE) depend -j $(nproc); $(MAKE) -j $(nproc)
	touch kaldi.done
endif

tensorflow.done:
	test -d tensorflow || git clone --depth=1 --branch=master https://github.com/tensorflow/tensorflow.git

# deps on the package, for kaldi input pipeline and kaldi io utils
espnet.done:
	test -d espnet || git clone --depth=1 https://github.com/espnet/espnet.git
	cd espnet && git fetch -t && git checkout v.0.5.3
	if [ -f kaldi.done ] && [ -s kaldi ]; then ln -s `readlink -f kaldi` espnet/tools/kaldi; elif [ -f kaldi.done ] && [ ! -s kaldi ]; then ln -s kaldi espnet/tools/kaldi; fi
	touch espnet.done

# google wordpiece subword
sentencepiece.done:
	test -d sentencepiece || git clone --depth=1 https://github.com/google/sentencepiece.git
	cd sentencepiece && mkdir build && cd build && (cmake3 .. || cmake ..) && $(MAKE)
	touch sentencepiece.done

# nmt 
moses.done:
	test -d moses || git clone --depth=1 https://github.com/moses-smt/mosesdecoder.git moses
	touch moses.done

# Chinese segment word 
cppjieba.done:
	test -d cppjieba || git clone --depth=1 https://github.com/applenob/cppjieba.git cppjieba
	touch cppjieba.done

check_install:
	$(PYTHON) install/check_install.py

clean: clean_extra clean_espnet deltann-clean
	rm -rf kaldi 
	rm -rf espnet
	rm -rf *.done
	find . -iname "*.pyc" -delete

clean_espnet:
	-rm espnet/tools/kaldi

clean_python:
	find . -iname "*.pyc" -delete

clean_extra:
	rm -rf nkf.done sentencepiece.done mecab.done swig.done moses.done
	rm -rf nkf sentencepiece mecab swig moses


# deltann
# make under `tensorflow/tensorflow:devel-py3` docker for cpu
deltann: apt.done yaml.done abseil.done jemalloc.done tcmalloc.done \
	tensorflow-src.done jsoncpp.done curl.done https_client.done go.done

apt.done:
	cmake --version || if ! [ -f /etc/centos-release ]; then $(SUDO) apt-get update && $(SUDO) $(APT_INSTALL) autoconf libtool cmake; else $(SUDO) yum install -y cmake3; fi
	$(PIP_INSTALL) --user shyaml meson ninja
	touch apt.done

abseil.done:
	test -d abseil-cpp || git clone --depth=1 https://github.com/abseil/abseil-cpp.git
	test -d googletest || git clone --depth=1 https://github.com/google/googletest.git
	touch abseil.done

tensorflow-src.done:
	# copy tensorflow src from docker
	if ! [ -d tensorflow ]; then rm -rf tensorflow && $(SUDO) cp -r /tensorflow_src tensorflow && $(SUDO) chown $(shell id -u):$(shell id -g) tensorflow; fi
	touch tensorflow-src.done

yaml.done:
	rm -rf yaml
	mkdir -p yaml
	cd yaml;git clone --depth=1 https://github.com/jbeder/yaml-cpp.git
	cd yaml/yaml-cpp; git fetch -t; git checkout yaml-cpp-0.6.0; export CXXFLAGS="-fPIC"; export CFLAGS="-fPIC"; cmake .; $(MAKE) prefix=.
	touch yaml.done

jemalloc.done:
	touch jemalloc.done

tcmalloc.done:
	touch tcmalloc.done

jsoncpp.done:
	if ! [ -d jsoncpp ]; then git clone --depth=1 https://github.com/open-source-parsers/jsoncpp.git; fi
	cd jsoncpp; export PATH=~/.local/bin:$(PATH); CXX=g++ BUILD_TYPE=release LIB_TYPE=shared .travis_scripts/meson_builder.sh
	touch jsoncpp.done

mbedtls.done:
	if ! [ -d mbedtls ]; then git clone --depth=1 --recursive https://github.com/ARMmbed/mbedtls.git; fi
	cd mbedtls && make clean
	cd mbedtls && SHARED=1 CLAFGS='-O2 -Werror -fPIC' make && $(SUDO) make install
	touch mbedtls.done

curl.done: mbedtls.done
	# https://github.com/curl/curl/blob/master/docs/INSTALL.md#select-tls-backend
	if ! [ -d curl ]; then git clone --depth=1 https://github.com/curl/curl.git; fi
	cd curl && ./buildconf && ./configure --without-ssl --with-mbedtls && make -j && $(SUDO) make install
	touch curl.done

https_client.done:
	if ! [ -d https_client ];then git clone https://github.com/HISONA/https_client.git; fi
	cd https_client && make
	touch https_client.done

go.done:
	cd install && ./install-go.sh
	touch go.done

deltann-clean:
	-rm -rf *.done jsoncpp yaml tensorflow abseil-cpp curl mbedtls .cache/
