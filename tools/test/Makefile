.PHONY: all clean
ifeq ($(origin MAKEFILE_DIR), undefined) 
MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
endif 
TOOLSDIR :=$(realpath $(CURDIR)/..)
MAINROOT :=$(realpath $(TOOLSDIR)/..)
GTEST_DIR :=$(TOOLSDIR)/googletest/googletest


CXX := g++ 
NVCC := nvcc
PYTHON_BIN_PATH= python
CC :=
AR := ar
CXXFLAGS := -O2 --std=c++11 -pthread
LDFLAGS :=
STDLIB :=


INC := \
  -I$(TOOLSDIR)/abseil-cpp \
  -I$(TOOLSDIR)/googletest/googletest/include \
  -I$(MAINROOT) \

TF_LIBS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
LIB := $(TF_LIBS) -ldl

SRCS := \
  test_main.cc \

HOST_ARCH=x86_64
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

GENDIR := $(MAKEFILE_DIR)/gen/
TGTDIR := $(GENDIR)$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(TGTDIR)obj/
BINDIR := $(TGTDIR)bin/
LIBDIR := $(TGTDIR)lib/

#TEST_OBJ := $(addprefix $(OBJDIR), $(patsubst %.cc, $(OBJDIR)%.o, $(SRCS)))
TEST_OBJ := $(patsubst %.cc, $(OBJDIR)%.o, $(SRCS))
# gtest lib
GTEST_LIB := $(LIBDIR)libgtest.a
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
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@ $(GTEST_LIB) $(LIB)

clean:
	-rm -rf gen
	-rm $(BIN)
