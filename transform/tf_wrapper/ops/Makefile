# Find where we're running from, so we can store generated files here.

ifeq ($(origin MAKEFILE_DIR), undefined) 
MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
MAIN_ROOT := $(realpath $(MAKEFILE_DIR)/../)
endif 

#$(info $(MAKEFILE_DIR))
#$(info $(MAIN_ROOT))

CXX := g++-7
NVCC := nvcc
PYTHON_BIN_PATH= python3
CC :=
AR :=
CXXFLAGS :=
LDFLAGS :=
STDLIB :=

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		HOST_OS := linux
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := ios
	endif
endif

#HOST_ARCH := $(shell if [[ $(shell uname -m) =~ i[345678]86 ]]; then echo x86_32; else echo $(shell uname -m); fi)
HOST_ARCH=x86_64
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

GENDIR := $(MAKEFILE_DIR)/gen/
TGTDIR := $(GENDIR)$(TARGET)_$(TARGET_ARCH)/
OBJDIR := $(TGTDIR)obj/
BINDIR := $(TGTDIR)bin/
LIBDIR := $(TGTDIR)lib/


TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
# Fix TF LDFLAGS issue on macOS.
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' | sed "s/-l:libtensorflow_framework.1.dylib/-ltensorflow_framework.1/")
#TF_INCLUDES := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIBS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CXXFLAGS += -fPIC -shared -O2 -std=c++11 -DFEATURE_VERSION=\"$(shell git rev-parse --short HEAD)\" $(TF_CFLAGS)
INCLUDES := -I$(MAIN_ROOT) \
			-I$(MAIN_ROOT)/ops/op_kernels \
			-I$(MAIN_ROOT)/ops \
			-I$(MAIN_ROOT)/ops/cppjieba/deps \
			-I$(MAIN_ROOT)/ops/cppjieba/include \
			-I$(MAIN_ROOT)/ops/op_kernels \
			-I$(MAIN_ROOT)/../kernels/add_rir_noise_aecres \
			-I$(MAIN_ROOT)/../kernels

LDFLAGS += $(TF_LFLAGS) 

CORE_CC_EXCLUDE_SRCS := \
$(wildcard op_kernels/*test.cc) \
$(wildcard op_kernels/*test_util.cc)

# src and tgts
LIB_SRCS_ALL := $(wildcard op_kernels/*.cc) \
                $(wildcard ../../kernels/*.cc) \
                $(wildcard ../../kernels/add_rir_noise_aecres/*.cpp)
LIB_SRCS := $(filter-out $(CORE_CC_EXCLUDE_SRCS), $(LIB_SRCS_ALL))
LIB_OBJS := $(addprefix $(OBJDIR), $(patsubst %.cc, %.o, $(patsubst %.cpp, %.o, $(LIB_SRCS))))

# lib
SHARED_LIB := x_ops.so

TEST_SRC := $(wildcard op_kernels/*_test.cc)
TEST_OBJ := $(addprefix $(OBJDIR), $(patsubst %.cc, $(OBJS_DIR)%.o, $(TEST_SRC)))
TEST_BIN := $(addprefix $(BINDIR), $(patsubst %.cc, $(OBJS_DIR)%.bin, $(TEST_SRC)))
#TEST_BIN := $(BINDIR)test

all: $(SHARED_LIB) $(TEST_BIN)

$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ $(LDFLAGS)

$(SHARED_LIB): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) -fPIC -shared -o $@ $^ $(STDLIB) $(LDFLAGS)

$(STATIC_LIB): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) crsv $@ $^

${TEST_BIN}: $(TEST_OBJ) $(STATIC_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) $^ -o $@ $(STATIC_LIB)

.PHONY: clean
clean:
	-rm -rf $(GENDIR)
	-rm -f x_ops.*so
