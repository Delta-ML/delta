# Settings for Linux.
ifeq ($(TARGET), linux)
    CXX := g++
    CC := gcc
    AR := ar
    CXXFLAGS += -fPIC
    CCFLAGS += -fPIC

    ifeq ($(ENGINE),TF)
        $(info this will compile with $(ENGINE))
        CXXFLAGS += -DUSE_TF
        TF_DIR := $(ROOT_DIR)/../tools/tensorflow
        TF_LIB_DIR := $(TF_DIR)/bazel-bin/tensorflow
        DPL_TF_LIB_DIR := $(TFLIBDIR)/tensorflow
        
       INCLUDES += -I$(TF_DIR) \
                -I$(TF_DIR)/bazel-genfiles \
                -I$(TF_DIR)/bazel-tensorflow/external/nsync/public/ \
                -I$(TF_DIR)/bazel-tensorflow \
                -I$(TF_DIR)/bazel-tensorflow/external/com_google_protobuf/src \
                -I$(TF_DIR)/bazel-tensorflow/external/eigen_archive

        # copy so
        TF_SO := $(TF_LIB_DIR)/libtensorflow_*.*
        $(shell mkdir -p $(DPL_TF_LIB_DIR) && cp -RP $(TF_SO) $(DPL_TF_LIB_DIR))

        LIBS += -L$(DPL_TF_LIB_DIR) -ltensorflow_cc -ltensorflow_framework

        ABSEIL := $(ROOT_DIR)/../tools/abseil-cpp
        INCLUDES += -I$(ABSEIL) 

    else ifeq ($(ENGINE),TFLITE)
        $(info this will compile with $(ENGINE))
        CXXFLAGS += -DUSE_TFLITE
        TFLITE_DIR := $(TF_DIR)
        TFLITE_LIB_DIR := $(TF_DIR)/bazel-bin/tensorflow/lite/experimental/c
        DPL_TFLITE_LIB_DIR := $(TFLIBDIR)/tflite

        # copy so
        TFLITE_SO := $(TFLITE_LIB_DIR)/libtensorflowlite_c.so
        $(shell mkdir -p $(DPL_TFLITE_LIB_DIR) && cp $(TFLITE_SO) $(DPL_TFLITE_LIB_DIR))

        INCLUDES += -I$(TFLITE_DIR) -I$(TF_DIR)/bazel-tensorflow/external/com_google_protobuf/src
        LIBS += -L$(DPL_TFLITE_LIB_DIR) -ltensorflowlite_c

    else ifeq ($(ENGINE),TFSERVING)
        $(info this will compile with $(ENGINE))
				HTTPS_CLIENT :=$(ROOT_DIR)../tools/https_client
        CXXFLAGS += -DUSE_TF_SERVING -fpermissive
				INCLUDES += -I$(HTTPS_CLIENT)

        LIBS += -L$(HTTPS_CLIENT)/mbedtls/library  -lmbedx509 -lmbedtls -lmbedcrypto
        #LIBS += -lcurl
    else
        $(warning not support engine $(ENGINE))
    endif
endif

