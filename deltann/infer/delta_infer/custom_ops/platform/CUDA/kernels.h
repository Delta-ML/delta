#ifndef _DELTA_INFER_CUDA_KERNELS_H_
#define _DELTA_INFER_CUDA_KERNELS_H_

#include "delta_infer/core/config.h"
#include "delta_infer/core/debug.h"
#include "delta_infer/custom_ops/transformer_cell_functor.h"

namespace tensorflow {

namespace delta {

class CublasGemmAlgo : public Entry {
 public:
  CublasGemmAlgo(cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT)
      : Entry(), _algo(algo) {}

  ~CublasGemmAlgo() {}

  operator cublasGemmAlgo_t() { return _algo; }

 public:
  cublasGemmAlgo_t _algo;
};

} /* namespace delta */

template <typename T>
using DType = typename DeltaTraits<GPUDevice, T>::DataType;

template <typename T>
void MultiHeadAtentionLauncher(
    DType<T>* query, DType<T>* key, DType<T>* value, DType<T>* q_buf,
    DType<T>* k_buf, DType<T>* v_buf, DType<T>* qk_buf, DType<T>* transpose_dst,
    DType<T>* attr_out_buf, const DType<T> scaler, int batch_size,
    int from_seq_len, int head_num, int size_per_head, const DType<T>* bias_Q,
    const DType<T>* bias_K, const DType<T>* bias_V, const DType<int>* mask,
    cublasHandle_t cublas_handle, cudaStream_t stream);

template <typename T>
void add_bias_act_kernelLauncher(DType<T>* out, const DType<T>* bias, int m,
                                 int n, cudaStream_t stream);

template <typename T>
void add_bias_input_layernorm_kernelLauncher(DType<T>* out,
                                             const DType<T>* input_tensor,
                                             const DType<T>* bias,
                                             const DType<T>* gamma,
                                             const DType<T>* beta, int m, int n,
                                             cudaStream_t stream);

template <typename T>
void add_bias_input_jump_layernorm(DType<T>* out, const DType<T>* input_tensor,
                                   const DType<T>* jump, const DType<T>* bias,
                                   const DType<T>* gamma, const DType<T>* beta,
                                   int m, int n, cudaStream_t stream);

template <typename T>
void rel_shift(DType<T>* BD, DType<T>* BD_BUF, int sqlen, int klen,
               int batch_size, int n_head, cudaStream_t stream);

template <typename T>
void add_bias_and_relu(DType<T>* attn_out_buff1,
                       const DType<T>* ff_layer_1_bias, int m, int n,
                       cudaStream_t stream);

template <typename T>
void add_input_layernorm(DType<T>* out, const DType<T>* input_tensor,
                         const DType<T>* gamma, const DType<T>* beta, int m,
                         int n, cudaStream_t stream);

template <typename T>
void combine_bias(const DType<T>* bias1_tmp, const DType<T>* ff_layer_2_bias,
                  int d_model, DType<T>* ff_kernel_bias_combine);

template <typename T>
void transpose_head_and_seq(DType<T>* data, int sqlen, int n_head, int d_head,
                            DType<T>* out, cudaStream_t stream);

template <typename T>
void transpose_head_num_and_seq(DType<T>* attn_vec, int sqlen, int bsz,
                                int n_head, int d_head, DType<T>* attn_vec_out,
                                cudaStream_t stream);

template <typename T>
void add_bias_and_split(DType<T>* w_heads, int mlen, int sqlen, int bsz,
                        int n_head, int d_head, const DType<T>* r_w_bias,
                        const DType<T>* r_r_bias, DType<T>* rw_head_q,
                        DType<T>* rr_head_q, DType<T>* w_head_k,
                        DType<T>* w_head_v, cudaStream_t stream);

template <typename T>
void attn_prob_softmax(DType<T>* ac, DType<T>* bd, const DType<T>* attn_mask,
                       int mlen, int sqlen, int bsz, int n_head, int d_head,
                       cudaStream_t stream);

} /* namespace tensorflow */

#endif
