
# DELTA-NN compile

Deltann support tensorflow， tensorflow lite，and tensorflow serving。

## Tensorflow C++
Build tensorflow for Linux :

1. Install deltann

```
cd tools/ && ./install/install-deltann.sh
```

2. Config tensorflow build
```
cd tools/tensorflow
```
Configure your system build by running the ./configure,


3. Build tensoflow library

**CPU-only**
```
bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so
```
**GPU support**

For GPU support, set cuda=Y during configuration and specify the versions of CUDA and cuDNN.
```
bazel build -c opt --config=cuda --verbose_failures //tensorflow:libtensorflow_cc.so
```
**Tensoflow TensorRT support**

For TensorRT support set Y during configuration and specify the location where TensorRT.

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
    path="/nfs/project/gaoyonghu/tools/android-ndk-r16b",
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
