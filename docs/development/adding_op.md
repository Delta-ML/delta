# Adding Tensorflow Op

All `custom-op` are under `delta/layers/ops/` directory.


## Eigen Tensor

[Eigen Tensor](https://github.com/eigenteam/eigen-git-mirror/blob/master/unsupported/Eigen/CXX11/src/Tensor/README.md) 
is unsupported eigen package, which is the underlying of 
[Tensorflow Tensor](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.h).


## Implement Op Kernel 

Implement your op kernel class for underlying computing.


## Create Tensorlow Op Wapper

Wapper the op kernel by Tensorflow Op or Tensorflow Lite Op. 


## Tensorflow

* [Guide for New Op](https://www.tensorflow.org/guide/extend/op)  
* [shape inference](https://github.com/tensorflow/tensorflow/core/framework/shape_inference.h)


## Tensorflow-Lite

* [TFLite custom ops](https://www.tensorflow.org/lite/guide/ops_custom).
* [TFLite select ops](https://www.tensorflow.org/lite/guide/ops_select)


## References
- [custom-op](https://github.com/tensorflow/custom-op)
- [lingvo](https://github.com/tensorflow/lingvo/tree/master/lingvo/core/ops)
- [Tensorflow Ops](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/kernels)
- [Tensorflow Lite Ops](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/kernels)
