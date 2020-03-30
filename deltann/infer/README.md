# Delta Inference Lib

> Ref to [coopper doc](http://cooper.didichuxing.com/docs/document/40259732)
>
> 注意： 目前构建在TF version 2.0 版本，,同时支持1.x版本. (已测试版本tf v1.14.0 ~ v2.0.0)。


## Compile

```bash
$ conda activate tf-2.0 # 也就是说，需要安装tf 2.0 的环境下完成编译
(tf-2.0)$ mkdir build & cd build
(tf-2.0)$ cmake .. & make -j8
(tf-2.0)$ ls delta_infer/ # 生成插件库地址
libdelta_infer.so 
(tf-2.0)$ ./delta_infer/test/basic_test # 测试 custom grappler
Get registered optimizer: test_optimizer suc!     
```

## Python API

### 直接安装
```bash
$ cd delta_infer/python
$ python setup.py install 
```

### wheel 打包
```bash
$ cd delta_infer/python
$ python setup.py bdist_wheel
$ pip install dist/delta_infer-0.0.1-cp36-cp36m-linux_x86_64.whl
```

## 问题

~~目前已知的一个问题，公版的tf2.0 wheel安装包是有问题的，包内部缺少`tensorflow/core/protobuf`的内部cpp文件内容, 最终会导致`undefined`连接错误。~~

~~而我们按照wheel包的编译方式从源码重新编译发现，没有上述问题，所以有理由怀疑是开源版本的安装包在安装过程中剪裁了部分内容，或者是其它问题，暂不可知。~~


