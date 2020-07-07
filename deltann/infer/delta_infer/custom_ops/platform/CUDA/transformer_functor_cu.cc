#include "delta_infer/custom_ops/platform/CUDA/cuda_checks.h"
#include "delta_infer/custom_ops/platform/CUDA/kernels.h"
#include "delta_infer/custom_ops/transformer_cell_functor.h"

namespace tensorflow {

namespace delta {

void load_config(const char* path = "./gemm_config.in") {
  const int GEMM_NUM = 5;
  int cublasAlgo[GEMM_NUM];  // total five gemm
  auto read_cfg = [&]() {
    FILE* fd = fopen(path, "r");
    int err = 0;
    if (fd == NULL) {
      printf("gemm_config.in is not found, using default GEMM algorithm.\n");
      return false;
    } else {
      err = fscanf(fd, "%d%d%d%d%d", &cublasAlgo[0], &cublasAlgo[1],
                   &cublasAlgo[2], &cublasAlgo[3], &cublasAlgo[4]);
      for (int i = 0; i < GEMM_NUM; i++) {
        // add default gemm algo for i-th gemm
        auto algo_ptr = std::make_shared<CublasGemmAlgo>(
            CublasGemmAlgo(static_cast<cublasGemmAlgo_t>(cublasAlgo[i])));
        delta::Config::Instance().add(std::to_string(i), algo_ptr);
      }
      fclose(fd);
    }
    if (err != 5) {
      printf(
          "ERROR: loading GEMM algorithms from gemm_config.in, using default "
          "GEMM algorithms\n");
      return false;
    }
    return true;
  };
  if (!read_cfg()) {
    // TODO set default algo for gemm
    for (int i = 0; i < GEMM_NUM; i++) {
      // add default gemm algo for i-th gemm
      auto algo_ptr = std::make_shared<CublasGemmAlgo>(CublasGemmAlgo());
      delta::Config::Instance().add(std::to_string(i), algo_ptr);
    }
  }
}

} /* namespace delta */

template <>
TransformerCellFunctor<GPUDevice, float>::TransformerCellFunctor() {
  // load config from global at construction of TransformerCell
  delta::load_config();
}

template <>
void TransformerCellFunctor<GPUDevice, float>::init(
    TransformerParam<GPUDevice, float>& param) {
  int buf_size = param.batch_size * param.head_num * param.from_seq_len *
                 param.size_per_head;
  int qk_buf_size = param.batch_size * param.head_num * param.from_seq_len *
                    param.from_seq_len;
  if (_buf == nullptr) {
    _buf = _allc.malloc(buf_size * 7 + qk_buf_size + buf_size * 6);
    _query_buf = _buf;
    _key_buf = _buf + buf_size;
    _value_buf = _buf + 2 * buf_size;
    _q_buf = _buf + 3 * buf_size;
    _k_buf = _buf + 4 * buf_size;
    _v_buf = _buf + 5 * buf_size;
    _qk_buf = _buf + 6 * buf_size;
    _transpose_dst = _qk_buf + qk_buf_size;
    // for feed forward
    _attr_out_buf = _transpose_dst + buf_size;
    _attr_matmul_buf = _attr_out_buf + buf_size;
    _inter_matmul_buf = _attr_matmul_buf + buf_size;
  }
}

template <>
TransformerCellFunctor<GPUDevice, float>::~TransformerCellFunctor() {
  if (_buf) {
    _allc.free(_buf);
    _buf = nullptr;
    _query_buf = nullptr;
    _key_buf = nullptr;
    _value_buf = nullptr;
    _q_buf = nullptr;
    _k_buf = nullptr;
    _v_buf = nullptr;
    _qk_buf = nullptr;
    _transpose_dst = nullptr;
    _attr_out_buf = nullptr;
    _attr_matmul_buf = nullptr;
    _inter_matmul_buf = nullptr;
  }
}

int check = 0;

template <>
void TransformerCellFunctor<GPUDevice, float>::operator()(
    OpKernelContext* context, TransformerParam<GPUDevice, float>& param) {
  int m = param.batch_size * param.from_seq_len;
  int k = param.head_num * param.size_per_head;
  int n = k;
  typename _traits::DataType alpha = (typename _traits::DataType)1.0f;
  typename _traits::DataType beta = (typename _traits::DataType)0.0f;
  DELTA_SCOPE {
    auto sum = CheckSum<GPUDevice, float>(param.from_tensor);
    printf("didi 0 ck sum: %lf %d \n", sum, check++);
  };
  cublas(GemmEx(param.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                param.kernel_Q, _traits::ComputeType, n, param.from_tensor,
                _traits::ComputeType, k, &beta, _query_buf,
                _traits::ComputeType, n, _traits::ComputeType,
                *static_cast<delta::CublasGemmAlgo*>(
                    delta::Config::Instance()["0"].get())));
  DELTA_SCOPE {
    auto sum = CheckSum<GPUDevice, float>(_query_buf);
    printf("didi 1 ck sum: %lf\n", sum);
  };

  cublas(GemmEx(param.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                param.kernel_K, _traits::ComputeType, n, param.to_tensor,
                _traits::ComputeType, k, &beta, _key_buf, _traits::ComputeType,
                n, _traits::ComputeType,
                *static_cast<delta::CublasGemmAlgo*>(
                    delta::Config::Instance()["0"].get())));

  cublas(GemmEx(param.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                param.kernel_V, _traits::ComputeType, n, param.to_tensor,
                _traits::ComputeType, k, &beta, _value_buf,
                _traits::ComputeType, n, _traits::ComputeType,
                *static_cast<delta::CublasGemmAlgo*>(
                    delta::Config::Instance()["0"].get())));
  typename _traits::DataType scaler = 1 / sqrtf(param.size_per_head * 1.0f);

  MultiHeadAtentionLauncher<float>(
      _query_buf, _key_buf, _value_buf, _q_buf, _k_buf, _v_buf, _qk_buf,
      _transpose_dst, _attr_out_buf, scaler, param.batch_size,
      param.from_seq_len, param.head_num, param.size_per_head, param.bias_Q,
      param.bias_K, param.bias_V, param.mask, param.cublas_handle,
      param.stream);
#if 1
  DELTA_SCOPE {
    auto sum = CheckSum<GPUDevice, float>(_attr_out_buf);
    printf("didi _attr_out_buf before ck sum: %lf\n", sum);
    sum = CheckSum<GPUDevice, float>(param.att_output_kernel);
    printf("didi param.att_output_kernel before ck sum: %lf\n", sum);
  };

  // feed forward
  cublas(GemmEx(param.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                param.att_output_kernel, _traits::ComputeType, n, _attr_out_buf,
                _traits::ComputeType, k, &beta, _attr_matmul_buf,
                _traits::ComputeType, n, _traits::ComputeType,
                *static_cast<delta::CublasGemmAlgo*>(
                    delta::Config::Instance()["0"].get())));

  DELTA_SCOPE {
    auto sum = CheckSum<GPUDevice, float>(_attr_matmul_buf);
    printf("didi _attr_matmul_buf after gemm ck sum: %lf\n", sum);
  };

  add_bias_input_layernorm_kernelLauncher<float>(
      _attr_matmul_buf, param.from_tensor, param.att_output_bias,
      param.att_output_layernorm_gamma, param.att_output_layernorm_beta, m, n,
      param.stream);

  DELTA_SCOPE {
    auto sum = CheckSum<GPUDevice, float>(_attr_matmul_buf);
    printf("didi _attr_matmul_buf after addbias_laynorm kernel ck sum: %lf\n",
           sum);
  };

  n *= 4;
  cublas(GemmEx(param.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                param.inter_kernel, _traits::ComputeType, n, _attr_matmul_buf,
                _traits::ComputeType, k, &beta, _inter_matmul_buf,
                _traits::ComputeType, n, _traits::ComputeType,
                *static_cast<delta::CublasGemmAlgo*>(
                    delta::Config::Instance()["1"].get())));

  add_bias_act_kernelLauncher<float>(_inter_matmul_buf, param.inter_bias, m, n,
                                     param.stream);

  DELTA_SCOPE {
    auto sum = CheckSum<GPUDevice, float>(_inter_matmul_buf);
    printf("didi _inter_matmul_buf after gemm and addbias kernel ck sum: %lf\n",
           sum);
  };

  n = k;
  k *= 4;
  cublas(GemmEx(param.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                param.output_kernel, _traits::ComputeType, n, _inter_matmul_buf,
                _traits::ComputeType, k, &beta, param.transformer_out,
                _traits::ComputeType, n, _traits::ComputeType,
                *static_cast<delta::CublasGemmAlgo*>(
                    delta::Config::Instance()["2"].get())));

  add_bias_input_layernorm_kernelLauncher<float>(
      param.transformer_out, _attr_matmul_buf, param.output_bias,
      param.output_layernorm_gamma, param.output_layernorm_beta, m, n,
      param.stream);
  DELTA_SCOPE {
    auto sum = CheckSum<GPUDevice, float>(param.transformer_out);
    printf(
        "didi param.transformer_out after gemm and addbias kernel ck sum: "
        "%lf\n",
        sum);
  };

    // TODO
#endif
}

} /* namespace tensorflow */
