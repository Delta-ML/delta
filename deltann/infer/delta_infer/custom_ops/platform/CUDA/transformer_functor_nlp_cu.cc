#include "delta_infer/custom_ops/transformer_cell_nlp_functor.h"
#include "delta_infer/custom_ops/platform/CUDA/cuda_checks.h"
#include "delta_infer/custom_ops/platform/CUDA/kernels.h"

namespace tensorflow {

namespace delta {

} /* namespace delta */

template<>
TransformerCellNLPFunctor<GPUDevice, float>::TransformerCellNLPFunctor() {
}

template<>
void TransformerCellNLPFunctor<GPUDevice, float>::init(TransformerNLPParam<GPUDevice, float>& param) {
    if(!param.w_initialized) {
        typename _traits::DataType alpha = (typename _traits::DataType)1.0f;
        typename _traits::DataType beta = (typename _traits::DataType)0.0f;
        int rank = 75;
        int m = param.d_model;
        int k = rank * 3;
        int n = param.n_head * param.d_head * 3;
        // alloc combine
        _att_kernel_QKV_combine = _allc.malloc(m * n);
        // combine kernel of dense
        cublas(GemmEx(param.cublas_handle,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      n, m, k,
                      &alpha,
                      param.att_kernel_QKV, _traits::ComputeType, n,
                      param.att_kernel_QKV_1, _traits::ComputeType, k,
                      &beta,
                      _att_kernel_QKV_combine, _traits::ComputeType, n,
                      _traits::ComputeType,
                      CUBLAS_GEMM_DEFAULT));

        { // combine feed forward kernel into one
            // combine kernel
            _ff_kernel_combine = _allc.malloc(param.kernel1_dim * param.d_model);
            m = param.kernel1_dim;
            n = param.d_model;
            k = param.kernel2_dim;
            cublas(GemmEx(param.cublas_handle,
                          CUBLAS_OP_N, CUBLAS_OP_N,
                          n, m, k,
                          &alpha,
                          param.ff_layer_2_kernel, _traits::ComputeType, n,
                          param.ff_layer_2_1_kernel, _traits::ComputeType, k,
                          &beta,
                          _ff_kernel_combine, _traits::ComputeType, n,
                          _traits::ComputeType,
                          CUBLAS_GEMM_DEFAULT));
            // combine bias
            float* bias_tmp = _allc.malloc(param.d_model);
            m = 1;
            n = param.d_model;
            k = param.kernel2_dim;
            cublas(GemmEx(param.cublas_handle,
                          CUBLAS_OP_N, CUBLAS_OP_N,
                          n, m, k,
                          &alpha,
                          param.ff_layer_2_kernel, _traits::ComputeType, n,
                          param.ff_layer_2_1_bias, _traits::ComputeType, k,
                          &beta,
                          bias_tmp, _traits::ComputeType, n,
                          _traits::ComputeType,
                          CUBLAS_GEMM_DEFAULT));
            // combine all bias
            _ff_kernel_bias_combine = _allc.malloc(param.d_model);
            combine_bias<float>(bias_tmp, param.ff_layer_2_bias, param.d_model, _ff_kernel_bias_combine);

            // free all 
            _allc.free(bias_tmp);
        }

        param.w_initialized = true;
        // alloc buf
        _cat = _allc.malloc( (param.mlen + param.sqlen) * param.d_model * param.batch_size);
        _w_heads = _allc.malloc((param.mlen + param.sqlen) * param.batch_size * param.n_head * param.d_head * 3);
        _r_head_k = _allc.malloc((param.mlen + param.sqlen) * param.n_head * param.d_head);
        _r_head_k_trans = _allc.malloc((param.mlen + param.sqlen) * param.n_head * param.d_head);
        _rw_head_q = _allc.malloc(param.sqlen * param.batch_size * param.n_head * param.d_head);
        _rr_head_q = _allc.malloc(param.sqlen * param.batch_size * param.n_head * param.d_head);
        _w_head_k = _allc.malloc((param.mlen + param.sqlen) * param.batch_size * param.n_head * param.d_head);
        _w_head_v = _allc.malloc((param.mlen + param.sqlen) * param.batch_size * param.n_head * param.d_head);
        _AC = _allc.malloc((param.mlen + param.sqlen) * param.sqlen * param.batch_size * param.n_head);
        _BD = _allc.malloc((param.mlen + param.sqlen) * param.sqlen * param.batch_size * param.n_head);
        _BD_BUF = _allc.malloc((param.mlen + param.sqlen) * param.sqlen * param.batch_size * param.n_head);
        _attn_vec = _allc.malloc(param.batch_size * param.n_head * param.sqlen * param.d_head);
        _attn_vec_out = _allc.malloc(param.batch_size * param.n_head * param.sqlen * param.d_head);
        _attn_out = _allc.malloc(param.sqlen * param.batch_size * param.d_model);
        _attn_out_buff1 = _allc.malloc(param.sqlen * param.batch_size * param.kernel1_dim);
        _attn_out_buff2 = _allc.malloc(param.sqlen * param.batch_size * param.d_model);
    }
}

template<>
TransformerCellNLPFunctor<GPUDevice, float>::~TransformerCellNLPFunctor() {
    if(_att_kernel_QKV_combine) {
        _allc.free(_att_kernel_QKV_combine);
        _allc.free(_cat);
        _allc.free(_w_heads);
        _allc.free(_r_head_k);
        _allc.free(_r_head_k_trans);
        _allc.free(_rw_head_q);
        _allc.free(_rr_head_q);
        _allc.free(_w_head_k);
        _allc.free(_w_head_v);
        _allc.free(_AC);
        _allc.free(_BD);
        _allc.free(_BD_BUF);
        _allc.free(_attn_vec);
        _allc.free(_attn_vec_out);
        _allc.free(_ff_kernel_combine);
        _allc.free(_ff_kernel_bias_combine);
        _allc.free(_attn_out);
        _allc.free(_attn_out_buff1);
        _allc.free(_attn_out_buff2);
    }
}

template<>
void TransformerCellNLPFunctor<GPUDevice, float>::operator() (OpKernelContext* context, TransformerNLPParam<GPUDevice, float>& param) {
    // concat
    _copy_d2d.async(_cat, param.mems, param.mlen * param.batch_size * param.d_model, param); 
    _copy_d2d.async(_cat + param.mlen * param.batch_size * param.d_model, param.w, param.sqlen * param.batch_size * param.d_model, param); 

    // compute _w_head
    typename _traits::DataType alpha = (typename _traits::DataType)1.0f;
    typename _traits::DataType beta = (typename _traits::DataType)0.0f;
#if 0
    int rank = 75;
    int m = (param.mlen + param.sqlen) * param.batch_size;
    int n = rank * 3;
    int k = param.d_model;
    float* tmp = _allc.malloc(m * n);
    cublas(GemmEx(param.cublas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  n, m, k,
                  &alpha,
                  param.att_kernel_QKV_1, _traits::ComputeType, n,
                  _cat, _traits::ComputeType, k,
                  &beta,
                  tmp, _traits::ComputeType, n,
                  _traits::ComputeType,
                  CUBLAS_GEMM_DEFAULT));
    DELTA_SCOPE{
        CheckElts<GPUDevice, float>(_cat, "_cat");
        CheckElts<GPUDevice, float>(tmp, "tmp");
    };
    m = m; 
    n = param.n_head * param.d_head * 3;
    k = rank * 3;
    cublas(GemmEx(param.cublas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  n, m, k,
                  &alpha,
                  param.att_kernel_QKV, _traits::ComputeType, n,
                  tmp, _traits::ComputeType, k,
                  &beta,
                  _w_heads, _traits::ComputeType, n,
                  _traits::ComputeType,
                  CUBLAS_GEMM_DEFAULT));
    DELTA_SCOPE{
        CheckElts<GPUDevice, float>(_cat, "_cat");
        CheckElts<GPUDevice, float>(param.mems, "param.mems");
        CheckElts<GPUDevice, float>(param.w, "param.w");
        CheckElts<GPUDevice, float>(_w_heads, "_w_heads");
    };


#else
    int m = (param.mlen + param.sqlen) * param.batch_size;
    int n = param.n_head * param.d_head * 3;
    int k = param.d_model;
    cublas(GemmEx(param.cublas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  n, m, k,
                  &alpha,
                  _att_kernel_QKV_combine, _traits::ComputeType, n,
                  _cat, _traits::ComputeType, k,
                  &beta,
                  _w_heads, _traits::ComputeType, n,
                  _traits::ComputeType,
                  CUBLAS_GEMM_DEFAULT));
    DELTA_SCOPE{
        CheckElts<GPUDevice, float>(_att_kernel_QKV_combine, "_att_kernel_QKV_combine");
        CheckElts<GPUDevice, float>(_cat, "_cat");
        CheckElts<GPUDevice, float>(param.mems, "param.mems");
        CheckElts<GPUDevice, float>(param.w, "param.w");
        CheckElts<GPUDevice, float>(_w_heads, "_w_heads");
    };
#endif

    m = param.mlen + param.sqlen;
    n = param.n_head * param.d_head;
    k = param.d_model;
    cublas(GemmEx(param.cublas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  n, m, k,
                  &alpha,
                  param.att_kernel_r, _traits::ComputeType, n,
                  param.pos_emb, _traits::ComputeType, k,
                  &beta,
                  _r_head_k, _traits::ComputeType, n,
                  _traits::ComputeType,
                  CUBLAS_GEMM_DEFAULT));

    // transpose
    transpose_head_and_seq<float>(_r_head_k, (param.mlen + param.sqlen), param.n_head, param.d_head, _r_head_k_trans, param.stream);

    add_bias_and_split<float>(_w_heads, 
                       param.mlen, param.sqlen, param.batch_size, param.n_head, param.d_head,
                       param.r_w_bias, param.r_r_bias,
                       _rw_head_q, _rr_head_q, 
                       _w_head_k, _w_head_v,
                       param.stream);

    m = param.sqlen;
    n = (param.mlen + param.sqlen);
    k = param.d_head;
#if (CUDART_VERSION >= 10000)
    cublas(GemmStridedBatchedEx(param.cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                n, m, k, 
                                &alpha, 
                                _w_head_k, traits::ComputeType, k, n * k, 
                                _rw_head_q, traits::ComputeType, k, m * k,
                                &beta, 
                                _AC, traits::ComputeType, n, m * n, 
                                param.batch_size * param.n_head,
                                traits::ComputeType, 
                                CUBLAS_GEMM_DEFAULT));
#else
    cublas(SgemmStridedBatched(param.cublas_handle,
                               CUBLAS_OP_T, CUBLAS_OP_N,
                               n, m, k,
                               &alpha, 
                               _w_head_k, k, n * k, 
                               _rw_head_q, k, m * k,
                               &beta, 
                               _AC, n, m * n, 
                               param.batch_size * param.n_head));  
#endif
    DELTA_SCOPE{
        CheckElts<GPUDevice, float>(_rw_head_q, "_rw_head_q");
        CheckElts<GPUDevice, float>(_AC, "_AC");
    };


    m = param.sqlen;
    n = (param.mlen + param.sqlen);
    k = param.d_head;
    for(int batch_idx = 0; batch_idx < param.batch_size; batch_idx++) {
#if (CUDART_VERSION >= 10000)
        cublas(GemmStridedBatchedEx(param.cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    n, m, k, 
                                    &alpha, 
                                    _r_head_k_trans, traits::ComputeType, k, n * k, 
                                    _rr_head_q + batch_idx * m * k * param.n_head, traits::ComputeType, k, m * k,
                                    &beta, 
                                    _BD + batch_idx * m * n * param.n_head, traits::ComputeType, n, m * n, 
                                    param.n_head,
                                    traits::ComputeType, 
                                    CUBLAS_GEMM_DEFAULT));
#else
        cublas(SgemmStridedBatched(param.cublas_handle,
                                   CUBLAS_OP_T, CUBLAS_OP_N,
                                   n, m, k,
                                   &alpha, 
                                   _r_head_k_trans, k, n * k, 
                                   _rr_head_q + batch_idx * m * k * param.n_head, k, m * k,
                                   &beta, 
                                   _BD + batch_idx * m * n * param.n_head, n, m * n, 
                                   param.n_head));  
#endif
    }
    DELTA_SCOPE{
        CheckElts<GPUDevice, float>(_BD, "_BD before rel_shift", 20);
        CheckElts<GPUDevice, float>(_AC, "before softmax _AC", 20);//(param.mlen + param.sqlen) * param.sqlen * param.batch_size * param.n_head);
    };

    rel_shift<float>(_BD, _BD_BUF, param.sqlen, param.mlen + param.sqlen, param.batch_size, param.n_head, param.stream);
    DELTA_SCOPE{
        CheckElts<GPUDevice, float>(_BD_BUF, "_BD_BUF", 20);
    };

    attn_prob_softmax<float>(_AC,
                             _BD_BUF, 
                             param.attn_mask, 
                             param.mlen, param.sqlen, param.batch_size, param.n_head, param.d_head,
                             param.stream);

    DELTA_SCOPE{
        CheckElts<GPUDevice, float>(_AC, "_AC", 20);//(param.mlen + param.sqlen) * param.sqlen * param.batch_size * param.n_head);
    };

    m = param.sqlen;
    n = param.d_head;
    k = (param.mlen + param.sqlen);
#if (CUDART_VERSION >= 10000)
    cublas(GemmStridedBatchedEx(param.cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n, m, k, 
                                &alpha, 
                                _w_head_v, traits::ComputeType, n, n * k, 
                                _AC, traits::ComputeType, k, m * k,
                                &beta, 
                                _attn_vec, traits::ComputeType, n, m * n, 
                                param.batch_size * param.n_head,
                                traits::ComputeType, 
                                CUBLAS_GEMM_DEFAULT));
#else
    cublas(SgemmStridedBatched(param.cublas_handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               n, m, k,
                               &alpha, 
                               _w_head_v, n, n * k, 
                               _AC, k, m * k,
                               &beta, 
                               _attn_vec, n, m * n, 
                               param.batch_size * param.n_head));  
#endif
    transpose_head_num_and_seq<float>(_attn_vec, param.sqlen, param.batch_size, param.n_head, param.d_head, _attn_vec_out, param.stream);
    DELTA_SCOPE{
        CheckElts<GPUDevice, float>(_attn_vec_out, "_attn_vec_out", 12);//param.batch_size * param.n_head * param.sqlen * param.d_head);
        double sum = CheckSum<GPUDevice, float>(_attn_vec_out, param.batch_size * param.n_head * param.sqlen * param.d_head); 
        printf("_attn_vec_out check sum: %f\n", sum);
        sum = CheckSum<GPUDevice, float>(_attn_vec, param.batch_size * param.n_head * param.sqlen * param.d_head);
        printf("_attn_vec check sum: %f\n", sum);
        sum = CheckSum<GPUDevice, float>(_w_head_v, (param.mlen + param.sqlen) * param.batch_size * param.n_head * param.d_head);
        printf("_w_head_v check sum: %f\n", sum);
        sum = CheckSum<GPUDevice, float>(_AC, (param.mlen + param.sqlen) * param.sqlen * param.batch_size * param.n_head);
        printf("_AC check sum: %f\n", sum);
    };

    m = param.sqlen * param.batch_size;
    n = param.d_model;
    k = param.d_head * param.n_head;
    // _attn_vec(sqlen, batch_size, d_model) is attn_out
    cublas(GemmEx(param.cublas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  n, m, k,
                  &alpha,
                  param.att_kernel_o, _traits::ComputeType, param.d_model,
                  _attn_vec_out, _traits::ComputeType, k,
                  &beta,
                  _attn_out, _traits::ComputeType, n,
                  _traits::ComputeType,
                  CUBLAS_GEMM_DEFAULT));
    DELTA_SCOPE{
        CheckElts<GPUDevice, float>(_attn_out, "before norm _attn_out");
    };
    // layer norm, _attn_out is the final output
    add_input_layernorm<float>(_attn_out, param.w, 
                               param.att_layernorm_gamma, param.att_layernorm_beta, 
                               m, n, 
                               param.stream);
    DELTA_SCOPE{
        CheckElts<GPUDevice, float>(_attn_out, "_attn_out");
    };


    //  feed forward
    //  combine all three matmul
    m = param.sqlen * param.batch_size;
    n = param.kernel1_dim;
    k = param.d_model;
    cublas(GemmEx(param.cublas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  n, m, k,
                  &alpha,
                  param.ff_layer_1_kernel, _traits::ComputeType, n,
                  _attn_out, _traits::ComputeType, k,
                  &beta,
                  _attn_out_buff1, _traits::ComputeType, n,
                  _traits::ComputeType,
                  CUBLAS_GEMM_DEFAULT));
    // add bias and relu
    add_bias_and_relu<float>(_attn_out_buff1, param.ff_layer_1_bias, m, n, param.stream);

    m = param.sqlen * param.batch_size;
    n = param.d_model;
    k = param.kernel1_dim;
    cublas(GemmEx(param.cublas_handle,
                  CUBLAS_OP_N, CUBLAS_OP_N,
                  n, m, k,
                  &alpha,
                  _ff_kernel_combine, _traits::ComputeType, n,
                  _attn_out_buff1, _traits::ComputeType, k,
                  &beta,
                  _attn_out_buff2, _traits::ComputeType, n,
                  _traits::ComputeType,
                  CUBLAS_GEMM_DEFAULT));
    
    DELTA_SCOPE{
        CheckElts<GPUDevice, float>(_attn_out_buff2, "_attn_out_buff2");
    };
    add_bias_input_jump_layernorm<float>(param.transformer_out, 
                                         _attn_out_buff2, 
                                         _attn_out,
                                         _ff_kernel_bias_combine, 
                                         param.ff_layernorm_gamma, 
                                         param.ff_layernorm_beta, 
                                         m, n, 
                                         param.stream);
    DELTA_SCOPE{
        CheckElts<GPUDevice, float>(param.transformer_out, "param.transformer_out");
    }; 
}

} /* namespace tensorflow */ 
