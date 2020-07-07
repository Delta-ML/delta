#ifndef _DELTA_INFER_CUSTOM_OPS_UTILS_H_
#define _DELTA_INFER_CUSTOM_OPS_UTILS_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#ifdef WITH_CUDA
#include "delta_infer/custom_ops/platform/CUDA/cuda_checks.h"
#endif

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

enum class DeltaOpType { INT32, FP32, HALF };

template <typename Device, typename T>
struct DeltaTraits;

#ifdef WITH_CUDA
template <>
struct DeltaTraits<GPUDevice, float> {
  typedef float DataType;
  static const DeltaOpType OpType = DeltaOpType::FP32;
  static const cudaDataType_t ComputeType = CUDA_R_32F;
};
template <>
struct DeltaTraits<GPUDevice, int> {
  typedef int DataType;
  static const DeltaOpType OpType = DeltaOpType::INT32;
  static const cudaDataType_t ComputeType = CUDA_R_32F;
};
#endif

template <>
struct DeltaTraits<CPUDevice, float> {
  typedef float DataType;
  static const DeltaOpType OpType = DeltaOpType::FP32;
  typedef float ComputeType;
};

template <>
struct DeltaTraits<CPUDevice, int> {
  typedef int DataType;
  static const DeltaOpType OpType = DeltaOpType::INT32;
  typedef float ComputeType;
};

/*template<>
struct DeltaTraits<Eigen::half> {
    // __half from cuda define
    typedef __half DataType;
    static const DeltaOpType OpType = DeltaOpType::HALF;
};*/

} /* namespace tensorflow */

#endif
