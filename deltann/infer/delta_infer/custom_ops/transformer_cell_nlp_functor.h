#ifndef _DELTA_INFER_CUSTOM_OPS_TRANSFORMER_CELL_NLP_FUNCTOR_H_
#define _DELTA_INFER_CUSTOM_OPS_TRANSFORMER_CELL_NLP_FUNCTOR_H_

#define EIGEN_USE_GPU  /// very important for Eigen when using gpu stream

#include "delta_infer/core/debug.h"
#include "delta_infer/custom_ops/alloc.h"
#include "delta_infer/custom_ops/ops_utils.h"

namespace tensorflow {

template <typename Device, typename T>
class TransformerNLPParam {
 public:
  int batch_size{-1};
  int sqlen{-1};
  int mlen{-1};
  int d_model{-1};
  int n_head{-1};
  int d_head{-1};
  int kernel1_dim{-1};
  int kernel2_dim{-1};

 public:
  bool w_initialized{false};
  const T* r_w_bias;
  const T* r_r_bias;
  const T* w;
  const T* attn_mask;
  const T* mems;
  const T* pos_emb;
  const T* att_kernel_QKV;
  const T* att_kernel_QKV_1;
  const T* att_kernel_r;
  const T* att_kernel_o;
  const T* att_layernorm_gamma;
  const T* att_layernorm_beta;
  const T* ff_layernorm_gamma;
  const T* ff_layernorm_beta;
  const T* ff_layer_1_kernel;
  const T* ff_layer_1_bias;
  const T* ff_layer_2_kernel;
  const T* ff_layer_2_bias;
  const T* ff_layer_2_1_kernel;
  const T* ff_layer_2_1_bias;
  T* transformer_out;

  void init(OpKernelContext* context) {}
};

template <typename T>
class TransformerNLPParam<GPUDevice, T> {
 public:
  int batch_size{-1};
  int sqlen{-1};
  int mlen{-1};
  int d_model{-1};
  int n_head{-1};
  int d_head{-1};
  int kernel1_dim{-1};
  int kernel2_dim{-1};

 public:
  bool w_initialized{false};
  const T* r_w_bias;
  const T* r_r_bias;
  const T* w;
  const T* attn_mask;
  const T* mems;
  const T* pos_emb;
  const T* att_kernel_QKV;
  const T* att_kernel_QKV_1;
  const T* att_kernel_r;
  const T* att_kernel_o;
  const T* att_layernorm_gamma;
  const T* att_layernorm_beta;
  const T* ff_layernorm_gamma;
  const T* ff_layernorm_beta;
  const T* ff_layer_1_kernel;
  const T* ff_layer_1_bias;
  const T* ff_layer_2_kernel;
  const T* ff_layer_2_bias;
  const T* ff_layer_2_1_kernel;
  const T* ff_layer_2_1_bias;
  T* transformer_out;
#ifdef WITH_CUDA
  // extra parameters
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
  bool handle_initialized{false};

  void init(OpKernelContext* context) {
    if (!handle_initialized) {
      cublas(Create(&cublas_handle));
      stream = context->eigen_device<GPUDevice>().stream();
      cublas(SetStream(cublas_handle, stream));
      handle_initialized = true;
    }
  }
#endif
};

template <typename Device, typename T>
struct TransformerCellNLPFunctor {
  TransformerCellNLPFunctor();

  void init(TransformerNLPParam<Device, T>& param);

  ~TransformerCellNLPFunctor();

  void operator()(OpKernelContext* context,
                  TransformerNLPParam<Device, T>& param);

 private:
  T* _att_kernel_QKV_combine{
      nullptr};  // combine att_kernel_QKV with att_kernel_QKV_1
  T* _ff_kernel_combine{nullptr};       // combine all feed forward kernel
  T* _ff_kernel_bias_combine{nullptr};  // combine all feed forward bias
  T* _cat{nullptr};
  T* _w_heads{nullptr};
  T* _r_head_k{nullptr};
  T* _r_head_k_trans{nullptr};
  T* _rw_head_q{nullptr};
  T* _rr_head_q{nullptr};
  T* _w_head_k{nullptr};
  T* _w_head_v{nullptr};
  T* _AC{nullptr};
  T* _BD{nullptr};
  T* _BD_BUF{nullptr};
  T* _attn_vec{nullptr};
  T* _attn_vec_out{nullptr};
  T* _attn_out{nullptr};
  T* _attn_out_buff1{nullptr};
  T* _attn_out_buff2{nullptr};
  DeltaAlloc<Device, T> _allc;
  CopyToDevice<Device, float, T> _copy_d2d;
  typedef DeltaTraits<Device, T> _traits;
};

} /* namespace tensorflow */

#endif
