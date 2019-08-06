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

## Run under docker image tensorflow/tensorflow:latest-py3
SHELL = /bin/bash

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

basic: linux.done python.done espnet.done nlp_zh compile_ops.done

speech: kaldi.done

nlp_zh: cppjieba.done

extra: nkf.done sentencepiece.done mecab.done moses.done

PIP_INSTALL=pip3 --no-cache-dir install
APT_INSTALL=apt-get install -y --no-install-recommends 

PYTHON=python3

ci.done:
	pushd ci && /bin/bash build.sh && popd
	touch ci.done

PROTOC_ZIP := protoc-3.7.1-linux-x86_64.zip
protoc:
	$(SUDO) apt-get update && $(SUDO) $(APT_INSTALL) curl unzip
	curl -OL --retry 100 https://github.com/google/protobuf/releases/download/v3.7.1/$(PROTOC_ZIP) &&\
	$(SUDO) unzip -o $(PROTOC_ZIP) -d /usr/local bin/protoc &&\
	rm -f $(PROTOC_ZIP)

# only for test cpp
test.deps:
#	test -d tensorflow || git clone --depth=1 --branch=master https://github.com/tensorflow/tensorflow.git
	test -d abseil-cpp || git clone --depth=1 https://github.com/abseil/abseil-cpp.git
	test -d googletest || git clone --depth=1 https://github.com/google/googletest.git 

test: protoc test.deps
	make -C test

# install python packages
python.done:
	$(PIP_INSTALL) --user -r requirements.txt
	if [ -f /etc/lsb-release ]; then $(SUDO) $(APT_INSTALL) libsndfile1; elif [[ "$$OSTYPE" == "darwin"* ]]; then brew install libsndfile; else $(SUDO) yum install -y libsndfile-devel; fi;
	touch python.done

linux.done:
	if [ -f /etc/lsb-release ]; then $(SUDO) apt-get update && $(SUDO) $(APT_INSTALL) bc sox; else $(SUDO) yum install -y bc sox; fi;
	touch linux.done

# compile custom ops on your environment
compile_ops.done:
	/bin/bash -c 'pushd ../delta/layers/ops; bash build.sh delta'
	touch compile_ops.done

# only for speech
ifneq ($(strip $(KALDI)),)
kaldi.done:
	ln -s $(abspath $(KALDI)) kaldi
	touch kaldi.done
else
kaldi.done:
	test -d kaldi || git clone --depth=1 https://github.com/kaldi-asr/kaldi.git
	cd kaldi/tools; $(MAKE) all
	cd kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	touch kaldi.done
endif

# deps on the package, for kaldi input pipeline and kaldi io utils
espnet.done:
	test -d espnet || git clone --depth=1 https://github.com/espnet/espnet.git
	cd espnet && git fetch -t && git checkout v.0.4.0
	if [ -f kaldi.done ] && [ -s kaldi ]; then ln -s `readlink -f kaldi` espnet/tools/kaldi; elif [ -f kaldi.done ] && [ ! -s kaldi ]; then ln -s kaldi espnet/tools/kaldi; fi
	touch espnet.done

nkf.done:
	rm -rf nkf
	mkdir -p nkf
	cd nkf; wget https://ja.osdn.net/dl/nkf/nkf-2.1.4.tar.gz
	cd nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.
	touch nkf.done

# google wordpiece subword
sentencepiece.done:
	test -d sentencepiece || git clone --depth=1 https://github.com/google/sentencepiece.git
	cd sentencepiece && mkdir build && cd build && (cmake3 .. || cmake ..) && $(MAKE)
	touch sentencepiece.done

# nmt 
moses.done:
	test -d moses || git clone --depth=1 https://github.com/moses-smt/mosesdecoder.git moses
	touch moses.done

# Japanese segment word 
mecab.done: espnet.done
	test -d mecab || git clone --depth=1 https://github.com/taku910/mecab.git
	cd mecab/mecab && ./configure --with-charset=utf8 --prefix=$(PWD)/mecab && $(MAKE) && $(MAKE) check && $(MAKE) install
	cd mecab/mecab-ipadic && ./configure --with-charset=utf8 --with-mecab-config=$(PWD)/mecab/bin/mecab-config --prefix=$(PWD)/mecab && $(MAKE) && $(MAKE) install
	cd mecab && git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
	cd mecab/mecab-ipadic-neologd && export PATH=$(PWD)/mecab/bin:$(PATH) && ./bin/install-mecab-ipadic-neologd -n -y -p $(PWD)/mecab/mecab-ipadic-neologd
	$(PIP_INSTALL) mojimoji pykakasi
	if [ `$(PYTHON) --version | cut -c 8` -eq 3 ]; then \
		if [ ! -e swig.done ]; then \
			rm -rf swig; \
			mkdir -p swig; \
			cd swig; \
			wget https://sourceforge.net/projects/swig/files/swig/swig-3.0.12/swig-3.0.12.tar.gz; \
			tar zxvf swig-3.0.12.tar.gz; \
			cd swig-3.0.12 && ./configure prefix=$(PWD)/swig && $(MAKE) && $(MAKE) install; \
			touch $(PWD)/swig.done; \
		fi; \
		deactivate; \
		export PATH=$(PWD)/swig/bin:$(PWD)/mecab/bin:$(PATH); \
		export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH); \
		$(PIP_INSTALL) mecab-python3; \
	else \
		$(PIP_INSTALL) mecab-python; \
	fi
	touch mecab.done

# Chinese segment word 
cppjieba.done:
	test -d cppjieba || git clone --depth=1 https://github.com/yanyiwu/cppjieba.git cppjieba
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
	tensorflow-src.done jsoncpp.done curl.done https_client.done

apt.done:
	$(SUDO) $(APT_INSTALL) autoconf libtool
	cmake --version || if ! [ -f /etc/centos-release ]; then $(SUDO) $(APT_INSTALL) cmake; else $(SUDO) yum install -y cmake; fi

abseil.done: 
	test -d abseil-cpp || git clone --depth=1 https://github.com/abseil/abseil-cpp.git
	touch abseil.done

tensorflow-src.done:
	# copy tensorflow src from docker
	# if ! [ -d tensorflow ]; then $(SUDO) cp -r /tensorflow_src tensorflow && chown $(id -u):$(id -g) tensorflow; fi
	touch tensorflow-src.done

yaml.done:
	rm -rf yaml
	mkdir -p yaml
	cd yaml;git clone --depth=1 https://github.com/jbeder/yaml-cpp.git
	cd yaml/yaml-cpp; git checkout yaml-cpp-0.6.0; export CXXFLAGS="-fPIC"; export CFLAGS="-fPIC"; cmake .; $(MAKE) prefix=.
	touch yaml.done

jemalloc.done:
	touch jemalloc.done

tcmalloc.done:
	touch tcmalloc.done

jsoncpp.done:
	if ! [ -d jsoncpp ]; then git clone --depth=1 https://github.com/open-source-parsers/jsoncpp.git; fi
	cd jsoncpp;  cmake . -DBUILD_SHARED_LIBS=ON; make
	touch jsoncpp.done

mbedtls.done:
	if ! [ -d mbedtls ]; then git clone --depth=1 --recursive https://github.com/ARMmbed/mbedtls.git; fi
	cd mbedtls && make clean && SHARED=1 CLAFGS='-O2 -Werror -fPIC' make && $(SUDO) make install
	touch mbedtls.done

curl.done: mbedtls.done
	# https://github.com/curl/curl/blob/master/docs/INSTALL.md#select-tls-backend
	if ! [ -d curl ]; then git clone --depth=1 https://github.com/curl/curl.git; fi
	cd curl && ./buildconf && ./configure --without-ssl --with-mbedtls && make -j && $(SUDO) make install
	touch curl.done

https_client.done:
	git clone https://github.com/HISONA/https_client.git
	cd https_client && make
	touch https_client.done

deltann-clean:
	-rm -rf *.done jsoncpp yaml tensorflow abseil-cpp curl mbedtls .cache/
