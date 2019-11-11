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
.PHONY: all clean

HOST_ARCH=x86_64
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

ifeq ($(origin MAKEFILE_DIR), undefined) 
MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
endif 

TOOLSDIR :=$(realpath $(CURDIR)/..)
MAINROOT :=$(realpath $(TOOLSDIR)/..)
GENDIR := $(MAKEFILE_DIR)/gen/
TGTDIR := $(GENDIR)$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(TGTDIR)obj/
BINDIR := $(TGTDIR)bin/
LIBDIR := $(TGTDIR)lib/

GTEST_DIR :=$(TOOLSDIR)/googletest/googletest
DELTANN_DIR :=$(MAINROOT)/deltann

CXX := g++ 
NVCC := nvcc
PYTHON_BIN_PATH= python
CC :=
AR := ar
CXXFLAGS := -O2 --std=c++11 -pthread
LDFLAGS :=
STDLIB :=

# gtest lib
GTEST_LIB := $(LIBDIR)libgtest.a
#TF_LIBS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_LIBS := -L$(TOOLSDIR)/tensorflow/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework
LIB := $(DELTANN_DIR)/.gen/lib/libdeltann.a $(GTEST_LIB) $(TF_LIBS) -ldl

INC := \
  -I$(TOOLSDIR)/abseil-cpp \
  -I$(TOOLSDIR)/googletest/googletest/include \
  -I$(DELTANN_DIR) \

SRCS := \
  test_main.cc \
  $(DELTANN_DIR)/core/shape_test.cc \

#TEST_OBJ := $(addprefix $(OBJDIR), $(patsubst %.cc, $(OBJDIR)%.o, $(SRCS)))
TEST_OBJ := $(patsubst %.cc, $(OBJDIR)%.o, $(SRCS))
BIN:=test_main.bin
all:$(BIN) 

$(OBJDIR)%.o:%.cc
	echo $(TEST_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@ $(LDFLAGS)

# https://github.com/google/googletest/blob/master/googletest/README.md
$(GTEST_LIB): $(TEST_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -isystem ${GTEST_DIR}/include -I${GTEST_DIR} -pthread -c ${GTEST_DIR}/src/gtest-all.cc -o $(OBJDIR)/gtest-all.o
	$(AR) -r $@ $(OBJDIR)/gtest-all.o

$(BIN): $(GTEST_LIB) $(TEST_OBJ) 
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(LIB)

clean:
	-rm -rf gen
	-rm $(BIN)
