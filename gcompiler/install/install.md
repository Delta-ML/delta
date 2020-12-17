# Install

1. clone tensorflow r2.0.3

2. check envs as follows: 
```
== check python ===================================================
python version: 3.6.6
python branch: 
python build version: ('default', 'Jun 28 2018 17:14:51')
python compiler version: GCC 7.2.0
python implementation: CPython


== check os platform ===============================================
os: Linux
os kernel version: #1 SMP Thu Nov 8 23:39:32 UTC 2018
os release version: 3.10.0-957.el7.x86_64
os platform: Linux-3.10.0-957.el7.x86_64-x86_64-with-debian-stretch-sid
linux distribution: ('debian', 'stretch/sid', '')
linux os distribution: ('debian', 'stretch/sid', '')
mac version: ('', ('', '', ''), '')
uname: uname_result(system='Linux', node='k8s-deploy-olbenz-1566810569345-7cc55777cd-mvdlp', release='3.10.0-957.el7.x86_64', version='#1 SMP Thu Nov 8 23:39:32 UTC 2018', machine='x86_64', processor='x86_64')
architecture: ('64bit', '')
machine: x86_64


== are we in docker =============================================
No

== compiler =====================================================
c++ (Ubuntu 7.4.0-1ubuntu1~16.04~ppa1) 7.4.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


== check pips ===================================================
numpy                         1.18.5
protobuf                      3.12.4
tensorflow                    2.0.3
tensorflow-estimator          2.0.1
tensorflow-model-optimization 0.5.0

== check for virtualenv =========================================
False

== tensorflow import ============================================
tf.version.VERSION = 2.0.3
tf.version.GIT_VERSION = v2.0.3-0-g295ad27
tf.version.COMPILER_VERSION = 7.4.0
```
== env ==========================================================
LD_LIBRARY_PATH /usr/local/cuda-10.0/extras/CUPTI/lib64/:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64::/usr/local/cuda-10.0/extras/CUPTI/lib64/
DYLD_LIBRARY_PATH is unset

== nvidia-smi ===================================================
Wed Dec 16 19:41:54 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P40           On   | 00000000:02:00.0 Off |                    0 |
| N/A   34C    P0    56W / 250W |   8399MiB / 22919MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla P40           On   | 00000000:03:00.0 Off |                    0 |
| N/A   21C    P8     9W / 250W |     16MiB / 22919MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla P40           On   | 00000000:83:00.0 Off |                    0 |
| N/A   35C    P0    56W / 250W |   8401MiB / 22919MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla P40           On   | 00000000:84:00.0 Off |                    0 |
| N/A   38C    P0    57W / 250W |   8407MiB / 22919MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+

== cuda libs  ===================================================
/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart.so.10.0.130
/usr/local/cuda-10.0/targets/x86_64-linux/lib/libcudart_static.a

== tensorflow installed from info ==================
Name: tensorflow
Version: 2.0.3
Summary: TensorFlow is an open source machine learning framework for everyone.
Home-page: https://www.tensorflow.org/
Author-email: packages@tensorflow.org
License: Apache 2.0
Location: /tmp-data/zhanghui/envs/venv/lib/python3.6/site-packages
Required-by: 

== python version  ==============================================
(major, minor, micro, releaselevel, serial)
(3, 6, 6, 'final', 0)

== bazel version  ===============================================
Build label: 0.26.1
Build time: Thu Jun 6 11:05:05 2019 (1559819105)
Build timestamp: 1559819105
Build timestamp as int: 1559819105
```

3. make sure bazel is 0.26.1 for tensoflow r2.0.3

4. compile tensorflow with CUDA 10.1

run `configure` to config tensorflow compiler options

```
bazel \
    --output_base /tmp/cache/tensorflow\
    build \
    --config=cuda \
    --config=nonccl \
    --config=noaws \
    --verbose_failures \
    //tensorflow:libtensorflow_cc.so \
    //tensorflow/tools/benchmark:benchmark_model \
    //tensorflow/tools/pip_package:build_pip_package


./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

pip install /tmp/tensorflow_pkg/tensorflow-2.0.3-cp36-cp36m-linux_x86_64.whl            
```

* check tensorflow install
do not check under tensorflow source dir:
```
(venv) luban@k8s-deploy-olbenz-1566810569345-7cc55777cd-mvdlp:/tmp-data/zhanghui/delta/tools/tensorflow$ python -c 'import tensorflow as tf; print(tf.__version__)'                      
Traceback (most recent call last):
  File "/tmp-data/zhanghui/delta/tools/tensorflow/tensorflow/python/platform/self_check.py", line 25, in <module>
    from tensorflow.python.platform import build_info
ImportError: cannot import name 'build_info'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/tmp-data/zhanghui/delta/tools/tensorflow/tensorflow/__init__.py", line 24, in <module>
    from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
  File "/tmp-data/zhanghui/delta/tools/tensorflow/tensorflow/python/__init__.py", line 49, in <module>
    from tensorflow.python import pywrap_tensorflow
  File "/tmp-data/zhanghui/delta/tools/tensorflow/tensorflow/python/pywrap_tensorflow.py", line 25, in <module>
    from tensorflow.python.platform import self_check
  File "/tmp-data/zhanghui/delta/tools/tensorflow/tensorflow/python/platform/self_check.py", line 27, in <module>
    raise ImportError("Could not import tensorflow. Do not import tensorflow "
ImportError: Could not import tensorflow. Do not import tensorflow from its source directory; change directory to outside the TensorFlow source tree, and relaunch your Python interpreter from there.
```

```
(venv) luban@k8s-deploy-olbenz-1566810569345-7cc55777cd-mvdlp:/tmp-data/zhanghui/delta/gcompiler$  python -c 'import tensorflow as tf; print(tf.__version__)'  
2.0.3
```

5. compile gcompiler

* Undefined reference to CheckOpMessageBuilder::NewString() when linking libtensorflow_cc.so
```
nm bazel-bin/tensorflow/libtensorflow_cc.so |c++filt | grep tensorflow::internal::CheckOpMessageBuilder::NewString
0000000009e29200 T tensorflow::internal::CheckOpMessageBuilder::NewString[abi:cxx11]()
```
CXX11_ABI Troubleshooting
If you get linker errors about undefined references to symbols that involve types in the std::__cxx11 namespace or the tag [abi:cxx11] then it probably indicates that you are trying to link together object files that were compiled with different values for the _GLIBCXX_USE_CXX11_ABI macro. This commonly happens when linking to a third-party library that was compiled with an older version of GCC. If the third-party library cannot be rebuilt with the new ABI then you will need to recompile your code with the old ABI.

- https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
- https://github.com/tensorflow/tensorflow/issues/5179
