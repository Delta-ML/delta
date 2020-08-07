#ifndef _DELTA_INFER_CUSTOM_OPS_TRANSFORMER_CELL_FUNCTOR_H_
#define _DELTA_INFER_CUSTOM_OPS_TRANSFORMER_CELL_FUNCTOR_H_

#define EIGEN_USE_GPU  /// very important for Eigen when using gpu stream

#include "delta_infer/core/debug.h"
#include "delta_infer/custom_ops/alloc.h"
#include "delta_infer/custom_ops/ops_utils.h"

namespace tensorflow {

template <typename Device, typename T>
class TransformerParam {
 public:
  int batch_size{-1};
  int from_seq_len{-1};
  int to_seq_len{-1};
  int head_num{-1};
  int size_per_head{-1};

 public:
  const T* from_tensor;
  const T* to_tensor;
  const T* kernel_Q;
  const T* kernel_K;
  const T* kernel_V;
  const T* bias_Q;
  const T* bias_K;
  const T* bias_V;
  const int* mask;
  const T* att_output_kernel;
  const T* att_output_bias;
  const T* att_output_layernorm_gamma;
  const T* att_output_layernorm_beta;
  const T* inter_kernel;
  const T* inter_bias;
  const T* output_kernel;
  const T* output_bias;
  const T* output_layernorm_gamma;
  const T* output_layernorm_beta;
  T* transformer_out;

  void init(OpKernelContext* context) {}
};

template <typename T>
class TransformerParam<GPUDevice, T> {
 public:
  int batch_size{-1};
  int from_seq_len{-1};
  int to_seq_len{-1};
  int head_num{-1};
  int size_per_head{-1};

 public:
  const T* from_tensor;
  const T* to_tensor;
  const T* kernel_Q;
  const T* kernel_K;
  const T* kernel_V;
  const T* bias_Q;
  const T* bias_K;
  const T* bias_V;
  const int* mask;
  const T* att_output_kernel;
  const T* att_output_bias;
  const T* att_output_layernorm_gamma;
  const T* att_output_layernorm_beta;
  const T* inter_kernel;
  const T* inter_bias;
  const T* output_kernel;
  const T* output_bias;
  const T* output_layernorm_gamma;
  const T* output_layernorm_beta;
  T* transformer_out;
#ifdef WITH_CUDA
  // extra parameters
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
  bool initialized{false};

  void init(OpKernelContext* context) {
    if (!initialized) {
      cublas(Create(&cublas_handle));
      stream = context->eigen_device<GPUDevice>().stream();
      cublas(SetStream(cublas_handle, stream));
      initialized = true;
    }
  }
#endif
};

template <typename Device, typename T>
struct TransformerCellFunctor {
  TransformerCellFunctor();

  void init(TransformerParam<Device, T>& param);

  ~TransformerCellFunctor();

  void operator()(OpKernelContext* context, TransformerParam<Device, T>& param);

 private:
  T* _buf{nullptr};
  T* _query_buf{nullptr};  /// used for swap space for first big gemm of query
  T* _key_buf{nullptr};    /// used for swap space for first big gemm of key
  T* _value_buf{nullptr};  /// used for swap space for first big gemm of value
  T* _q_buf{nullptr};      /// swap space of first big gemm of query with bias
  T* _k_buf{nullptr};      /// swap space of first big gemm of key with bias
  T* _v_buf{nullptr};      /// swap space of first big gemm of value with bias
  T* _qk_buf{nullptr};     // q matmu with k results
  T* _transpose_dst{nullptr};
  T* _attr_out_buf{nullptr};
  T* _attr_matmul_buf{nullptr};
  T* _inter_matmul_buf{nullptr};
  DeltaAlloc<Device, T> _allc;
  typedef DeltaTraits<Device, T> _traits;
};

} /* namespace tensorflow */

#endif
