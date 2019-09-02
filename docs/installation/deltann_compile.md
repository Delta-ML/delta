
# DELTA-NN compile

Deltann support tensorflow， tensorflow lite，and tensorflow serving.

## Tensorflow C++
Build tensorflow for Linux :

1. Install under deltann docker.

```
cd tools/ && ./install/install-deltann.sh
```

2. Config tensorflow build
```
cd tools/tensorflow
```
Configure your system build by running the ./configure,


3. Build tensoflow library

### CPU-only

```
bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so
```

mkl support 
```
bazel build -c opt --config=mkl --verbose_failures //tensorflow:libtensorflow_cc.so
```

### GPU support

Configure your system build by running the ./configure.

For GPU support, set cuda=Y during configuration and specify the versions of CUDA and cuDNN.
```
Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 10


Please specify the location where CUDA 10.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]:


Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
```

Build

```
bazel build -c opt --config=cuda --verbose_failures //tensorflow:libtensorflow_cc.so
```

### Tensoflow TensorRT support

Configure your system build by running the ./configure.
For TensorRT support, set Y during configuration and specify the versions of CUDA,  cuDNN, TensorRT, NCCL.

```
Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 10


Please specify the location where CUDA 10.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]:


Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:


Do you wish to build TensorFlow with TensorRT support? [y/N]: y
TensorRT support will be enabled for TensorFlow.

Please specify the location where TensorRT is installed. [Default is /usr/lib/x86_64-linux-gnu]:


Please specify the NCCL version you want to use. If NCCL 2.2 is not installed, then you can use version 1.3 that can be fetched automatically but it may have worse performance with multiple GPUs. [Default is 2.2]: 2.3


Please specify the location where NCCL 2 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:

```

set environmental variable

```
export TF_NEED_TENSORRT=1
```

Edit "tensorflow/BUILD", add the following code to tf_cc_shared_object of the file.

```
"//tensorflow/contrib/tensorrt:trt_engine_op_kernel",
"//tensorflow/contrib/tensorrt:trt_engine_op_op_lib",
```

```
sed -i '/\"\/\/tensorflow\/cc:cc_ops\",/a\"\/\/tensorflow\/contrib\/tensorrt:trt_engine_op_kernel\",\n\"\/\/tensorflow\/contrib\/tensorrt:trt_engine_op_op_lib\",' tensorflow/BUILD
```

Build

```
bazel build --config=opt --config=cuda  //tensorflow:libtensorflow_cc.so \
--action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
```

4. Build deltann

```
cd delta/deltann && ./build.sh linux x86_64 tf
```


## Tensorflow Lite
### Build tensorflow lite for arm.
1. Config ndk
Edit "tensorflow/WORKSPACE". Add the following code to the end of the file.

```
android_ndk_repository(
    name="androidndk",
    path="/ndk/path/android-ndk-r16b",
    api_level=21
)
```
2. Build tensoflow library

```
#armv7
bazel build -c opt --cxxopt=--std=c++11 \
    --config=android_arm //tensorflow/lite/experimental/c:libtensorflowlite_c.so

#arm64
bazel build -c opt --cxxopt=--std=c++11 \
    --config=android_arm64 //tensorflow/lite/experimental/c:libtensorflowlite_c.so
```

3. Build deltann

```
cd delta/deltann && ./build.sh android arm tflite
```

### Build TensorFlow lite for iOS

1. You need to run a shell script to download the dependencies you need:

```
tensorflow/lite/tools/make/download_dependencies.sh
```

2.  Build the library for all five supported architectures on iOS:

```
tensorflow/lite/tools/make/build_ios_universal_lib.sh
```
The resulting library is in tensorflow/lite/tools/make/gen/lib/libtensorflow-lite.a.


3. Build deltann

```
cd delta/deltann && ./build.sh ios arm tflite
```

### Tailor tensorflow lite library

There isn't an automatic way of doing this. You can edit tensorflow/lite/kernels/register.cc and tensorflow/lite/kernels/BUILD, delete some ops that you don't require.

Eg：delete lstm op if you don't require.

tensorflow/lite/kernels/register.cc：

```
--- a/tensorflow/lite/kernels/register.cc
+++ b/tensorflow/lite/kernels/register.cc
@@ -60,7 +60,6 @@ TfLiteRegistration* Register_BATCH_TO_SPACE_ND();
 TfLiteRegistration* Register_MUL();
 TfLiteRegistration* Register_L2_NORMALIZATION();
 TfLiteRegistration* Register_LOCAL_RESPONSE_NORMALIZATION();
 -TfLiteRegistration* Register_LSTM();
 TfLiteRegistration* Register_BIDIRECTIONAL_SEQUENCE_LSTM();
 TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
 TfLiteRegistration* Register_PAD();
@@ -184,7 +183,6 @@ BuiltinOpResolver::BuiltinOpResolver() {
   AddBuiltin(BuiltinOperator_L2_NORMALIZATION, Register_L2_NORMALIZATION());
   AddBuiltin(BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
              Register_LOCAL_RESPONSE_NORMALIZATION());
-  AddBuiltin(BuiltinOperator_LSTM, Register_LSTM(), /* min_version */ 1,
                /* max_version */ 2);
   AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM,
   Register_BIDIRECTIONAL_SEQUENCE_LSTM());
```

tensorflow/lite/kernels/BUILD：

```
--- a/tensorflow/lite/kernels/BUILD
+++ b/tensorflow/lite/kernels/BUILD
@@ -190,7 +190,6 @@ cc_library(
         "local_response_norm.cc",
         "logical.cc",
         "lsh_projection.cc",
-        "lstm.cc",
         "maximum_minimum.cc",
         "mfcc.cc",
         "mul.cc",
```

## Tensorflow Seving
1. Download tensorflow serving

```
git clone https://github.com/tensorflow/serving.git
cd serving
```

2. Build tensorflow serving

```
bazel build //tensorflow_serving/model_servers:tensorflow_model_server
```

3. Build deltann

```
cd delta/deltann && ./build.sh linux x86_64 tfserving
```
