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

# Find where we're running from, so we can store generated files here.
.PHONY: all examples clean

SHELL=/bin/bash

ifeq ($(origin MAKEFILE_DIR), undefined)
MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
endif

ROOT_DIR := $(MAKEFILE_DIR)/

HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = windows
else
UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
HOST_OS := linux
endif
ifeq ($(UNAME_S),Darwin)
	HOST_OS := osx
endif
endif

HOST_ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)


# platform
TARGET := $(HOST_OS)
# arch
TARGET_ARCH := $(HOST_ARCH)
# engine
ENGINE := TF
$(info $(origin ENGINE))
# debug mode
DEBUG := true

GENDIR := $(MAKEFILE_DIR)/.gen/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/

# dpl lib dir
TFLIBDIR := $(ROOT_DIR)/../dpl/lib

CC := gcc
CXX := g++
AR := ar

CFLAGS := -O2
CXXFLAGS := -std=c++11 -fPIC -DDELTANN_VERSION=\"$(shell git rev-parse --short HEAD)\"
LDFLAGS :=
ifeq ($(DEBUG), true)
  CXXFLAGS += -g
else
  CXXFLAGS += -DNDEBUG -O3 
  LDFLAGS += -Wl,--version-script=deltann_version_script.lds
endif
ARFLAGS := -r

YAML := $(ROOT_DIR)/../tools/yaml/yaml-cpp
JSONCPP := $(ROOT_DIR)/../tools/jsoncpp/
HTTP := $(ROOT_DIR)/../tools/https_client/
MBEDTLS := $(ROOT_DIR)/../tools/mbedtls

INCLUDES := -I$(ROOT_DIR) -I$(YAML)/include -I$(JSONCPP)/include -I$(HTTP) -I$(MBEDTLS)/include

LIBS := -lpthread -lstdc++ -ldl -L$(JSONCPP)/build-release_shared_g++/ -ljsoncpp

include $(MAKEFILE_DIR)/targets/${TARGET}_makefile.inc

export ROOT_DIR

export TARGET
export TARGET_ARCH
export ENGINE

export CC
export CXX
export AR

export GENDIR
export LIBDIR
export TFLIBDIR

export JSONCPP

export LIBS

TEST := $(wildcard core/*_test.cc core/utils/*_test.cc api/*_test.cc)
SRCS := $(filter-out $(TEST) , $(wildcard core/*.cc core/utils/*.cc api/*.cc))
OBJS := $(addprefix $(OBJDIR), $(patsubst %.cc, %.o,$(SRCS)))
STATIC_LIB := $(LIBDIR)libdeltann.a
SHARED_LIB := $(LIBDIR)libdeltann.so

TEST_SRC := test/test.cc
TEST_BIN := $(BINDIR)test

EXAMPLE_DIR := examples

all: $(STATIC_LIB) $(SHARED_LIB) $(OBJS) $(TEST_BIN)

$(STATIC_LIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(STATIC_LIB) $(OBJS)
	cp $(STATIC_LIB) $(MAKEFILE_DIR)/../dpl/lib/deltann

$(SHARED_LIB): $(OBJS) $(YAML)/libyaml-cpp.a
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -shared -fPIC -s -o $@ $(OBJS) $(YAML)/libyaml-cpp.a
	cp $(SHARED_LIB) $(MAKEFILE_DIR)/../dpl/lib/deltann

$(TEST_BIN): $(TEST_SRC)
	-@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TEST_SRC) -o $(TEST_BIN) -L $(LIBDIR) -ldeltann $(LIBS)

$(OBJDIR)%.o:%.cc
	-@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

examples:
	$(MAKE) -C $(EXAMPLE_DIR)

clean:
	-rm -rf $(GENDIR)
	$(MAKE) -C $(EXAMPLE_DIR) clean


