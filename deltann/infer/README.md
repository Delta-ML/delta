# Delta Inference Lib

Delta 推理架构基于tensorflow进行深度定制，依赖tensorflow本身的grappler以及custom ops机制完成高性能的图匹配逻辑和高性能算子库，`Delta Inference Lib`可以方便使用户在tensorflow内部完成大规模复杂子图替换，并采用已提供的高性能算子库支持模型在各个平台的高性能计算。

`Delta Inference Lib`核心依赖用户安装的tensorflow环境，用户并不需要从源码安装tensorflow，可以极大的减少使用成本。模型部署时，用户只需要把动态链接库一起发布即可。

`Delta Inference Lib`包含两部分核心组件：

* ***算法子图搜索引擎*** [Details](https://github.com/pangge/delta/blob/master/deltann/infer/docs/subgraphs.md)

* ***异构高性能算子库*** [Details](https://github.com/pangge/delta/blob/master/deltann/infer/docs/customops.md)

***<u>算法子图搜索引擎</u>*** 的核心逻辑是做子图匹配和替换，其支持用户使用任意tf语法定义的复杂算法结构，并根据用户训练好的模型，自动匹配这部分已定义算法并做替换，替换成用户设置的具体算子，最终完成大规模tensorflow子图结构的替换，从而极大的提升计算密集型和访存开销，为进一步性能提升打下坚实基础。

为了配合算法子图搜索引擎，`Delta Inference Lib`还提供了***<u>高性能的算子库</u>*** `delta_infer/custom_ops`, 高性算子库可以同时支持GPU、X86的高性能算子<sup>（未来可以扩展更多支持）</sup>，目前核心支持transformer模型的若干实现，核心支持硬件NV GPU平台 <sup> x86 后续会逐步添加</sup>。

## 编译

#### 编译custom_ops & delta_infer高性能库

```bash
$ conda activate tf-2.0 # 也就是说，需要安装tf 1.14(或者其它版本) 的环境下完成编译
(tf-1.14)$ mkdir build & cd build
(tf-1.14)$
(tf-1.14)$ cmake .. -DBUILD_DELTA_INFER_CUSTOM_OPS=ON -DBUILD_GPU=ON -DBUILD_WITH_EXAMPLE=ON
(tf-1.14)$ make -j$(nproc)
(tf-1.14)$ ls delta_infer/ # 生成插件库地址
libdelta_infer.so 
(tf-1.14)$ ls delta_infer/custom_ops/  # 高性能算子库
libcustom_ops.so
(tf-1.14)$ cd example/c++/ # 测试目录
```



## Python API

`Delta Inference Lib` 主要提供算法子图搜索引擎的Python API，安装方式如下；

#### 直接安装

```bash
$ cd delta_infer/python
$ python setup.py install 
```

#### wheel 打包
```bash
$ cd delta_infer/python
$ python setup.py bdist_wheel
$ pip install dist/delta_infer-0.0.1-cp36-cp36m-linux_x86_64.whl
```



## 注意

> 1. *目前已知的一个问题，公版的tf2.0 wheel安装包是有问题的，包内部缺少`tensorflow/core/protobuf`的内部cpp文件内容, 最终会导致`undefined`连接错误。而我们按照wheel包的编译方式从源码重新编译发现，没有上述问题，所以有理由怀疑是开源版本的安装包在安装过程中剪裁了部分内容。*
> 2. 目前构建在TF version 2.0 版本，,同时支持1.x版本. (已测试版本tf v1.14.0 ~ v2.0.0)。
>
> 2. 高性能算子库目前支持transformer模型，提供GPU高性能实现
> 3. 高性算子库部分代码参考了[Fast Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer)


