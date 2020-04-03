#include "delta_infer/custom_ops/platform/CUDA/kernels.h"

namespace tensorflow {

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T gelu(T x) {
    float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
    return x * cdf;
}

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;

  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);

  return val;
}


template <typename T>
__global__ void add_bias_act(T* out, const T* bias, int m, int n) {
    T val, reg_bias;

    int row_id = blockIdx.x;
    int ite = n / blockDim.x;
    int tid = threadIdx.x;

    for(int i = 0; i < ite; ++i)
    {
      reg_bias = __ldg(&bias[i * blockDim.x + tid]);
      row_id = blockIdx.x;

      while(row_id < m){
        val = out[tid + i * blockDim.x + row_id * n]+ reg_bias;
        out[tid + i * blockDim.x + row_id * n] = gelu<T>(val);
        row_id += gridDim.x;
      }
    }
}

template <typename T>
void add_bias_act_kernelLauncher(DType<T>* out, 
                                 const DType<T>* bias, 
                                 int m, int n, 
                                 cudaStream_t stream) {
    dim3 grid(m / 4);
    dim3 block(n / 4);
    assert(block.x <= 1024);
    add_bias_act<T><<<grid, block, 0, stream>>>(out, bias, m, n);
}

template void add_bias_act_kernelLauncher<float>(DType<float>* out, const DType<float>* bias, int m, int n, cudaStream_t stream);

template <typename T>
__global__ void add_bias_input_layernorm(T* out, 
                                         const T* input, 
                                         const T* bias, 
                                         const T* gamma, 
                                         const T* beta, 
                                         int m, int n) {
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean =  0.0f;
    float variance = 0.0f;

    float local_out = 0.0f;
    for(int i = tid; i < n; i += blockDim.x)
      local_out += (float)(out[blockIdx.x * n + i] + input[blockIdx.x * n + i] + __ldg(&bias[i]));

    mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
      s_mean = mean / n;
    __syncthreads();

    variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
    if(threadIdx.x == 0)
      s_variance = variance / n + 1e-30f;
    __syncthreads();

    for(int i = tid; i < n; i += blockDim.x)
      out[blockIdx.x * n + i] =
          (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
}

template <typename T> 
void add_bias_input_layernorm_kernelLauncher(DType<T>* out, 
                                             const DType<T>* input_tensor, 
                                             const DType<T>* bias, 
                                             const DType<T>* gamma, 
                                             const DType<T>* beta, 
                                             int m, int n, 
                                             cudaStream_t stream) {
    dim3 grid(m); 
    dim3 block(n); 
    assert(block.x <= 1024); 
    add_bias_input_layernorm<T><<<grid, block, 0, stream>>>(out, input_tensor, bias, gamma, beta, m, n);
}

template void add_bias_input_layernorm_kernelLauncher<float>(DType<float>* out, 
                                                      const DType<float>* input_tensor, 
                                                      const DType<float>* bias, 
                                                      const DType<float>* gamma, 
                                                      const DType<float>* beta, 
                                                      int m, int n, 
                                                      cudaStream_t stream);

template <typename T>
__global__ void add_bias_input_jump_layernorm_kernel(T* out, 
                                                     const T* input, 
                                                     const T* jump,
                                                     const T* bias, 
                                                     const T* gamma, 
                                                     const T* beta, 
                                                     int m, int n) {
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean =  0.0f;
    float variance = 0.0f;

    float local_out = 0.0f;
    for(int i = tid; i < n; i += blockDim.x)
      local_out += (float)(/*out[blockIdx.x * n + i] +*/ input[blockIdx.x * n + i] + jump[blockIdx.x * n + i] + __ldg(&bias[i]));

    mean = blockReduceSum<float>(local_out);
    if(threadIdx.x == 0)
      s_mean = mean / n;
    __syncthreads();

    variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
    if(threadIdx.x == 0)
      s_variance = variance / n + 1e-30f;
    __syncthreads();

    for(int i = tid; i < n; i += blockDim.x)
      out[blockIdx.x * n + i] =
          (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
}

template <typename T> 
void add_bias_input_jump_layernorm(DType<T>* out, 
                                   const DType<T>* input_tensor, 
                                   const DType<T>* jump,
                                   const DType<T>* bias, 
                                   const DType<T>* gamma, 
                                   const DType<T>* beta, 
                                   int m, int n, 
                                   cudaStream_t stream) {
    dim3 grid(m); 
    dim3 block(n); 
    assert(block.x <= 1024); 
    add_bias_input_jump_layernorm_kernel<T><<<grid, block, 0, stream>>>(out, input_tensor, jump, bias, gamma, beta, m, n);
}

template void add_bias_input_jump_layernorm<float>(DType<float>* out, 
                                                   const DType<float>* input_tensor, 
                                                   const DType<float>* jump,
                                                   const DType<float>* bias, 
                                                   const DType<float>* gamma, 
                                                   const DType<float>* beta, 
                                                   int m, int n, 
                                                   cudaStream_t stream);

template <typename T>
__global__ void rel_shift_kernel(T* bd, T* bd_buf, int sqlen, int klen) {
    T* start_addr = bd + blockIdx.x * sqlen * klen;
    T* start_addr_buf = bd_buf + blockIdx.x * sqlen * klen;
    T fetch_val;
    int ini_num = klen+1-sqlen;
    for(int i = threadIdx.x; i < sqlen*klen; i += blockDim.x) {
        int q_id = i / klen;
        int k_id = i % klen;
        int step = (ini_num + q_id) % klen;
        if(step != k_id) {
            start_addr_buf[i] = start_addr[i + sqlen -1 - q_id];
        }  else {
            start_addr_buf[i] = 0.0f;
        }
    }
}

template<typename T>
void rel_shift(DType<T>* BD, 
               DType<T>* BD_BUF,
               int sqlen, int klen, int batch_size, int n_head, 
               cudaStream_t stream) {
    int n = batch_size * n_head;
    dim3 grid(n);
    dim3 block(256);
    //printf("sqlen(%d), klen(%d), batch_size(%d), n_head(%d)\n ", sqlen, klen, batch_size, n_head);
    rel_shift_kernel<<<grid, block, 0, stream>>>(BD, BD_BUF, sqlen, klen);
}

template void rel_shift<float>(DType<float>* BD, DType<float>* BD_BUF, int sqlen, int klen, int batch_size, int n_head, cudaStream_t stream);

template <typename T>
__global__ void add_bias_and_relu_kernel(T* attn_out_buff, 
                                         const T* bias, 
                                         int m, int n) {
    for(int i = threadIdx.x; i < n; i += blockDim.x) {
        T val = (float)(attn_out_buff[blockIdx.x * n + i] + __ldg(&bias[i]));
        attn_out_buff[blockIdx.x * n + i] = val < 0.0f ? 0.0f : val;
    }
}

template <typename T>
void add_bias_and_relu(DType<T>* attn_out_buff1, const DType<T>* ff_layer_1_bias, 
                       int m, int n, 
                       cudaStream_t stream) {
    dim3 grid(m); 
    dim3 block(1024); 
    assert(block.x <= 1024); 
    add_bias_and_relu_kernel<<<grid, block, 0, stream>>>(attn_out_buff1, ff_layer_1_bias, m, n);
}

template void add_bias_and_relu<float>(DType<float>* attn_out_buff1, const DType<float>* ff_layer_1_bias, 
                                       int m, int n, cudaStream_t stream);


template <typename T>
__global__ void add_input_layernorm_kernel(T* out, 
                                           const T* input, 
                                           const T* gamma, 
                                           const T* beta, 
                                           int m, int n) {
    int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean =  0.0f;
    float variance = 0.0f;

    float local_out = 0.0f;
    for(int i = tid; i < n; i += blockDim.x) {
      local_out += (float)(out[blockIdx.x * n + i] + input[blockIdx.x * n + i]);
    }

    mean = blockReduceSum<float>(local_out);
    if(tid == 0)
      s_mean = mean / n;
    __syncthreads();

    variance = blockReduceSum<float>((local_out - s_mean) * (local_out - s_mean));
    if(tid == 0)
      s_variance = variance / n + 1e-30f;
    __syncthreads();

    for(int i = tid; i < n; i += blockDim.x)
      out[blockIdx.x * n + i] =
          (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[i])) + (float)(__ldg(&beta[i])));
}

template <typename T> 
void add_input_layernorm(DType<T>* out, 
                         const DType<T>* input_tensor, 
                         const DType<T>* gamma, 
                         const DType<T>* beta, 
                         int m, int n, 
                         cudaStream_t stream) {
    dim3 grid(m); 
    dim3 block(n); 
    assert(block.x <= 1024); 
    add_input_layernorm_kernel<T><<<grid, block, 0, stream>>>(out, input_tensor, gamma, beta, m, n);
}

template void add_input_layernorm<float>(DType<float>* out, 
                                         const DType<float>* bias, 
                                         const DType<float>* gamma, 
                                         const DType<float>* beta, 
                                         int m, int n, 
                                         cudaStream_t stream);

template <typename T>
__global__ void combine_bias_kernel(const T* bias1,
                                    const T* bias2,
                                    int n,
                                    T* out) {
    for(int i=threadIdx.x; i<n; i+=blockDim.x) {
        T bias1_val = __ldg(&bias1[i]);
        T bias2_val = __ldg(&bias2[i]);
        out[i] = bias1_val + bias2_val;
    }
}

template<typename T>
void combine_bias(const DType<T>* bias1_tmp, 
                  const DType<T>* ff_layer_2_bias, 
                  int d_model,  
                  DType<T>* ff_kernel_bias_combine) {
    dim3 grid(1);
    dim3 block(256);
    combine_bias_kernel<<<grid, block>>>(bias1_tmp, ff_layer_2_bias, d_model, ff_kernel_bias_combine);
}

template void combine_bias<float>(const DType<float>* bias1_tmp, 
                                  const DType<float>* ff_layer_2_bias, 
                                  int d_model,  
                                  DType<float>* ff_kernel_bias_combine);

} /* namespace tensorflow */

