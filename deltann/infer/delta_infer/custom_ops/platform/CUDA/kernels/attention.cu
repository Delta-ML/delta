#include "delta_infer/custom_ops/platform/CUDA/kernels.h"

namespace tensorflow {

#define FINAL_MASK 0xffffffff

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
__inline__ __device__ T warpReduceMax(T val) {
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

template<typename T>
__global__ void transpose(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head) { 
    int batch_id = blockIdx.x / (head_num * seq_len); 
    int seq_id = blockIdx.x % seq_len; 
    int head_id = (blockIdx.x % (head_num * seq_len))/ seq_len; 
    dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head 
        + head_id * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template <typename T>
__global__
void softmax_kernel(T* qk_buf, const int* attr_mask, const int batch_size, const int head_num, const int seq_len,
  const T scaler)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;
    int mask_offset = batch_id * seq_len * seq_len;

    __shared__ float s_sum, s_max;

    for(int i = 0; i < seq_len; ++i)
    {
      float qk = threadIdx.x < seq_len ? (float)qk_buf[threadIdx.x + qk_offset] : 0.0f;
      float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;

      mask_val = (1.0f - mask_val) * -10000.0f;

      float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val): -1e20f;

      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0)
      {
        s_sum = sum_val + 1e-30f;
      }
      __syncthreads();

      if(threadIdx.x < seq_len)
        qk_buf[threadIdx.x + qk_offset] = (T)(qk / s_sum);

      qk_offset += seq_len;
      mask_offset += seq_len;
    }
}

template <typename T>
__global__
void softmax_kernel_v2(T* qk_buf, const int* attr_mask, const int batch_size, const int head_num,
  const int seq_len, const float scaler)
{
    int batch_id = blockIdx.x / head_num / seq_len;
    int seq_id = blockIdx.x % seq_len;
    int qk_offset = blockIdx.x * seq_len;
    int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < seq_len ? (float)qk_buf[threadIdx.x + qk_offset] : 0.0f;
    float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;

    mask_val = (1.0f - mask_val) * -10000.0f;

    float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scaler + mask_val) : -1e20f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val + 1e-30f;
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

template<typename T>
__global__ void add_QKV_bias(T* Q, const T* bias_Q, 
                             T* K, const T* bias_K, 
                             T* V, const T* bias_V, 
                             T* q_buf, T* k_buf, T* v_buf, 
                             const int batch_size, const int seq_len, const int head_num, const int size_per_head, 
                             const int word_per_block) {
    T* data_ptr;
    T* buf_ptr;
    const T* bias_ptr;

    int m = batch_size * seq_len;
    int n = head_num * size_per_head;

    int qkv_id = blockIdx.x * word_per_block / m;
    int row_offset = (blockIdx.x * word_per_block % m) * n;

    if(qkv_id == 0) {
      data_ptr = Q + row_offset;
      buf_ptr = q_buf;
      bias_ptr = bias_Q;
    } else if(qkv_id == 1) {
      data_ptr = K + row_offset;
      buf_ptr = k_buf;
      bias_ptr = bias_K;
    } else {
      data_ptr = V + row_offset;
      buf_ptr = v_buf;
      bias_ptr = bias_V;
    }
#if 1
    // add bias and transpose
    int batch_id = (blockIdx.x * word_per_block % m) / seq_len;
    int head_id = threadIdx.x / size_per_head;
    int id_in_head = threadIdx.x % size_per_head;
    int word_start_id = (blockIdx.x * word_per_block) % seq_len;

    T bias = __ldg(&bias_ptr[threadIdx.x]);

    #pragma unroll
    for(int i = word_start_id; i < word_start_id + word_per_block; ++i) {
      T tmp = data_ptr[threadIdx.x] + bias;

      int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head +
        i * size_per_head + id_in_head;

      buf_ptr[target_id] = tmp;
      data_ptr += n;
    }
#else
    // error
    T bias = __ldg(&bias_ptr[threadIdx.x]);

    for(int i = 0; i < word_per_block; i++) {
        T tmp = data_ptr[threadIdx.x] + bias;
        int target_id = row_offset + i*n + threadIdx.x;
        buf_ptr[target_id] = tmp;
        data_ptr += n;
    }
#endif
}

template<typename T>
void MultiHeadAtentionLauncher(DType<T>* query,
                               DType<T>* key,
                               DType<T>* value,
                               DType<T>* q_buf, 
                               DType<T>* k_buf,
                               DType<T>* v_buf,
                               DType<T>* qk_buf,
                               DType<T>* transpose_dst,
                               DType<T>* attr_out_buf,
                               const DType<T> scaler,
                               int batch_size,
                               int from_seq_len,
                               int head_num,
                               int size_per_head,
                               const DType<T>* bias_Q,
                               const DType<T>* bias_K,
                               const DType<T>* bias_V,
                               const DType<int>* mask,
                               cublasHandle_t cublas_handle,
                               cudaStream_t stream) {
    int m = batch_size * from_seq_len; 
    int k = head_num * size_per_head;
    const int word_per_block = 1;
    assert(k <= 1024); 
    assert(m / word_per_block * 3 <= 65536); 
    dim3 grid(m / word_per_block * 3);
    dim3 block(k);
    DELTA_SCOPE{
        auto sum = CheckSum<GPUDevice, float>(query);
        printf("ccw query ck sum: %lf \n", sum);

        sum = CheckSum<GPUDevice, float>(key);
        printf("ccw key ck sum: %lf \n", sum);

        sum = CheckSum<GPUDevice, float>(value);
        printf("ccw value ck sum: %lf \n", sum);
    };

    add_QKV_bias<DType<T> ><<<grid, block, 0, stream>>>(query, bias_Q, 
                                                        key, bias_K, 
                                                        value, bias_V, 
                                                        q_buf, k_buf, v_buf, 
                                                        batch_size, from_seq_len, head_num, size_per_head, 
                                                        word_per_block);
    //cuda(PeekAtLastError());
    //cuda(DeviceSynchronize());
    DELTA_SCOPE{
        auto sum = CheckSum<GPUDevice, float>(q_buf);
        printf("ccw q_buf ck sum: %lf \n", sum);

        sum = CheckSum<GPUDevice, float>(k_buf);
        printf("ccw k_buf ck sum: %lf \n", sum);

        sum = CheckSum<GPUDevice, float>(v_buf);
        printf("ccw v_buf ck sum: %lf \n", sum);
    };

#if 1
    T alpha = 1.0f, beta = 0.0f;    
    typedef DeltaTraits<GPUDevice, T> traits;
    /// get _qk_buf[batch_size, head_num, from_seq_len, from_seq_len]
#if (CUDART_VERSION >= 10000)
    cublas(GemmStridedBatchedEx(cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                from_seq_len, from_seq_len, size_per_head, 
                                &alpha, 
                                k_buf, traits::ComputeType, size_per_head, from_seq_len * size_per_head, 
                                q_buf, traits::ComputeType, size_per_head, from_seq_len * size_per_head,
                                &beta, 
                                qk_buf, traits::ComputeType, from_seq_len, from_seq_len * from_seq_len, 
                                batch_size * head_num, 
                                traits::ComputeType, 
                                *static_cast<delta::CublasGemmAlgo*>(delta::Config::Instance()["3"].get())));  
#else
    cublas(SgemmStridedBatched(cublas_handle,
                               CUBLAS_OP_T, CUBLAS_OP_N,
                               from_seq_len, from_seq_len, size_per_head, 
                               &alpha, 
                               k_buf, size_per_head, from_seq_len * size_per_head, 
                               q_buf, size_per_head, from_seq_len * size_per_head,
                               &beta, 
                               qk_buf, from_seq_len, from_seq_len * from_seq_len, 
                               batch_size * head_num));  
#endif
    if(from_seq_len <= 32)
      block.x = 32;
    else if(from_seq_len > 32 && from_seq_len <= 64)
      block.x = 64;
    else if(from_seq_len > 64 && from_seq_len <= 128)
      block.x = 128;
    else if(from_seq_len > 128 && from_seq_len <= 256)
      block.x = 256;
    else if(from_seq_len > 256 && from_seq_len <= 512)
      block.x = 512;
    else
      block.x = 1024;
  
    DELTA_SCOPE{
        auto sum = CheckSum<GPUDevice, float>(qk_buf);
        printf("ccw qk_buf before softmax ck sum: %lf\n", sum);
    };
    if(batch_size * head_num <= 120) {
      grid.x = batch_size * head_num * from_seq_len;
      softmax_kernel_v2<DType<T> ><<<grid, block, 0, stream>>>(qk_buf, mask, batch_size, head_num, from_seq_len, scaler);
    } else {
      grid.x = batch_size * head_num;
      softmax_kernel<DType<T> ><<<grid, block, 0, stream>>>(qk_buf, mask, batch_size, head_num, from_seq_len, scaler);
    }  
    DELTA_SCOPE {
        auto sum = CheckSum<GPUDevice, float>(qk_buf);
        printf("ccw qk_buf after softmax ck sum: %lf\n", sum);
    };

#if (CUDART_VERSION >= 10000) /// cuda > 10.0
    cublas(GemmStridedBatchedEx(cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                size_per_head, from_seq_len, from_seq_len, 
                                &alpha, 
                                v_buf, traits::ComputeType, size_per_head, from_seq_len * size_per_head, 
                                qk_buf, traits::ComputeType, from_seq_len, from_seq_len * from_seq_len, 
                                &beta, 
                                transpose_dst, traits::ComputeType, size_per_head, from_seq_len * size_per_head, 
                                batch_size * head_num, 
                                traits::ComputeType, 
                                *static_cast<delta::CublasGemmAlgo*>(delta::Config::Instance()["4"].get())));      
#else
    cublas(SgemmStridedBatched(cublas_handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               size_per_head, from_seq_len, from_seq_len, 
                               &alpha, 
                               v_buf, size_per_head, from_seq_len * size_per_head, 
                               qk_buf, from_seq_len, from_seq_len * from_seq_len, 
                               &beta, 
                               transpose_dst, size_per_head, from_seq_len * size_per_head, 
                               batch_size * head_num));      
#endif
    DELTA_SCOPE{
        auto sum = CheckSum<GPUDevice, float>(transpose_dst);
        printf("ccw transpose_dst before softmax ck sum: %lf\n", sum);
    };
    const int seq_per_block = 1;
    grid.x = batch_size * head_num * from_seq_len / seq_per_block; 
    block.x = seq_per_block * size_per_head; 
    transpose<DType<T> ><<<grid, block, 0, stream>>>(transpose_dst, 
                                                     attr_out_buf, 
                                                     batch_size, 
                                                     from_seq_len, 
                                                     head_num, 
                                                     size_per_head);
    DELTA_SCOPE{
        auto sum = CheckSum<GPUDevice, float>(transpose_dst);
        printf("ccw transpose_dst after softmax ck sum: %lf\n", sum);
    };
#endif
}

template void MultiHeadAtentionLauncher<float>(DType<float>* query,
                                               DType<float>* key,
                                               DType<float>* value,
                                               DType<float>* q_buf, 
                                               DType<float>* k_buf,
                                               DType<float>* v_buf,
                                               DType<float>* qk_buf,
                                               DType<float>* transpose_dst,
                                               DType<float>* attr_out_buf,
                                               const DType<float> scaler,
                                               int batch_size,
                                               int from_seq_len, 
                                               int head_num, 
                                               int size_per_head,
                                               const DType<float>* bias_Q,
                                               const DType<float>* bias_K,
                                               const DType<float>* bias_V,
                                               const DType<int>* mask, 
                                               cublasHandle_t cublas_handle, 
                                               cudaStream_t stream);

template<typename T>
__global__ void transpose_head_and_seq_kernel(T* data, int sqlen, int n_head, int d_head, T* out) {
    extern __shared__ T smem[];
    int seq_id = blockIdx.x / n_head;
    int head_id = blockIdx.x % n_head;
    smem[threadIdx.x] = data[blockIdx.x*d_head + threadIdx.x];
    __syncthreads();
    out[head_id * sqlen * d_head + seq_id * d_head + threadIdx.x] = smem[threadIdx.x];
}

template<typename T>
void transpose_head_and_seq(DType<T>* data, int sqlen, int n_head, int d_head, DType<T>* out, cudaStream_t stream) {
    dim3 grid(sqlen * n_head);
    dim3 block(d_head);
    transpose_head_and_seq_kernel<<<grid, block, d_head*sizeof(T), stream>>>(data, sqlen, n_head, d_head, out);
}

template void transpose_head_and_seq<float>(DType<float>* data, int sqlen, int n_head, int d_head, DType<float>* out, cudaStream_t stream);

template<typename T>
__global__ void transpose_head_num_and_seq_kernel(T* attn_vec,
                                                  int sqlen, int bsz, int n_head, int d_head,
                                                  T* attn_vec_out) {
    extern __shared__ T smem[];
    int head_offset = blockIdx.x * d_head;
    int bsz_id = blockIdx.x / (n_head * sqlen);
    int head_id = blockIdx.x % (n_head * sqlen) / sqlen;
    int seq_id = blockIdx.x % sqlen;
    smem[threadIdx.x] = attn_vec[head_offset + threadIdx.x];
    __syncthreads();
    int head_len = n_head * d_head;
    int batch_len = head_len * bsz;
    attn_vec_out[seq_id * batch_len + bsz_id * head_len + head_id * d_head + threadIdx.x] = smem[threadIdx.x];
}

template<typename T>
void transpose_head_num_and_seq(DType<T>* attn_vec, 
                                int sqlen, int bsz, int n_head, int d_head,
                                DType<T>* attn_vec_out,
                                cudaStream_t stream) {
    dim3 grid(sqlen * bsz * n_head);
    dim3 block(d_head);
    transpose_head_num_and_seq_kernel<<<grid, block, d_head*sizeof(T), stream>>>(attn_vec, sqlen, bsz, n_head, d_head, attn_vec_out);
}

template void transpose_head_num_and_seq<float>(DType<float>* attn_vec, 
                                                int sqlen, int bsz, int n_head, int d_head,
                                                DType<float>* attn_vec_out,
                                                cudaStream_t stream);

template<typename T>
__global__ void add_bias_and_split_kernel(T* w_heads, 
                                          int mlen, int sqlen, int bsz, int n, int n_head, int d_head,
                                          const T* r_w_bias,
                                          const T* r_r_bias,
                                          T* rw_head_q, 
                                          T* rr_head_q, 
                                          T* w_head_k,
                                          T* w_head_v) {
    extern __shared__ T swap_space[];
    int start_idx = 0;
    if(blockIdx.y == 0) {
        // split and add bias
        int block_id = blockIdx.x - mlen * bsz;
        if(block_id >= 0) {
            start_idx = blockIdx.x * n * 3;
            for(int i=threadIdx.x; i<n; i+=blockDim.x) {
                swap_space[i] = w_heads[start_idx + i];
            }
            __syncthreads();
            int seq_id = block_id / bsz;
            int bsz_id = block_id % bsz;
            for(int i = 0; i < n_head; i++) {
                for(int j=threadIdx.x; j<d_head; j+=blockDim.x) {
                    T bias = __ldg(&r_w_bias[i*d_head + j]);
                    // from [seq, bsz, n_head, d_head] to [bsz , n_head, seq, d_head]
                    int out_id = bsz_id * n_head * sqlen * d_head + i * sqlen * d_head + seq_id * d_head + j;
                    rw_head_q[out_id] = swap_space[i*d_head + j] + bias;
                    bias = __ldg(&r_r_bias[i*d_head + j]);
                    rr_head_q[out_id] = swap_space[i*d_head + j] + bias;
                }
            }
        }
    } else if(blockIdx.y == 1) {
        // only split
        start_idx = blockIdx.x * n * 3 + n;
        for(int i=threadIdx.x; i<n; i+=blockDim.x) {
            swap_space[i] = w_heads[start_idx + i];
        }
        __syncthreads();
        int seq_id = blockIdx.x / bsz;
        int bsz_id = blockIdx.x % bsz;
        for(int i = 0; i < n_head; i++) {
           for(int j=threadIdx.x; j < d_head; j+=blockDim.x) {
               // from [seq, bsz, n_head, d_head] to [bsz , n_head, seq, d_head]
               int out_id = bsz_id * n_head * (mlen + sqlen) * d_head + i * (mlen + sqlen) * d_head + seq_id * d_head + j;
               w_head_k[out_id] = swap_space[i*d_head + j];;
           }
        }
    } else {
        // only split
        start_idx = blockIdx.x * n * 3 + 2 * n;
        for(int i=threadIdx.x; i<n; i+=blockDim.x) {
            swap_space[i] = w_heads[start_idx + i];
        }
        __syncthreads();
        int seq_id = blockIdx.x / bsz;
        int bsz_id = blockIdx.x % bsz;
        for(int i = 0; i < n_head; i++) {
           for(int j=threadIdx.x; j < d_head; j+=blockDim.x) {
               // from [seq, bsz, n_head, d_head] to [bsz , n_head, seq, d_head]
               int out_id = bsz_id * n_head * (mlen + sqlen) * d_head + i * (mlen + sqlen) * d_head + seq_id * d_head + j;
               w_head_v[out_id] = swap_space[i*d_head + j];;
           }
        }
    }
}

template <typename T> 
void add_bias_and_split(DType<T>* w_heads, 
                        int mlen, int sqlen, int bsz, int n_head, int d_head,
                        const DType<T>* r_w_bias,
                        const DType<T>* r_r_bias,
                        DType<T>* rw_head_q, 
                        DType<T>* rr_head_q, 
                        DType<T>* w_head_k, 
                        DType<T>* w_head_v,
                        cudaStream_t stream) {
    int m = (mlen + sqlen) * bsz;
    int n = n_head * d_head; // x 3
    assert(n <= 1024); 
    assert(m * 3 <= 65536); 
    dim3 grid(m, 3);
    dim3 block(128);
    add_bias_and_split_kernel<<<grid, block, n*sizeof(T), stream>>>(w_heads,
                                                                    mlen, sqlen, bsz, n, n_head, d_head,
                                                                    r_w_bias,
                                                                    r_r_bias,
                                                                    rw_head_q,
                                                                    rr_head_q,
                                                                    w_head_k,
                                                                    w_head_v);
};

template void add_bias_and_split<float>(DType<float>* w_heads, 
                                        int mlen, int sqlen, int bsz, int n_head, int d_head,
                                        const DType<float>* r_w_bias,
                                        const DType<float>* r_r_bias,
                                        DType<float>* rw_head_q, 
                                        DType<float>* rr_head_q, 
                                        DType<float>* w_head_k, 
                                        DType<float>* w_head_v,
                                        cudaStream_t stream);

#if 0
template<typename T>
__global__ void attn_prob_softmax_kernel(T* ac, 
                                         T* bd,
                                         const T* attn_mask, 
                                         const int sqlen, 
                                         const int klen, 
                                         const float scaler) {
    int input_offset = blockIdx.x * klen;
    int mask_offset = blockIdx.x % sqlen * klen;

    __shared__ float s_sum;

    float ac_val = threadIdx.x < klen  ? (float)ac[threadIdx.x + input_offset] : 0.0f;
    float bd_val = threadIdx.x < klen  ? (float)bd[threadIdx.x + input_offset] : 0.0f;
    float mask_val = threadIdx.x < klen ? (float)attn_mask[threadIdx.x + mask_offset] : 0.0f;

    float tmp = threadIdx.x < klen ? (ac_val + bd_val) * scaler * (1 - mask_val) - 1e30f * mask_val : 1e-30f;
    tmp = threadIdx.x < klen ? __expf((float)(tmp)) : 0.0f;
    float sum_val = blockReduceSum<float>(tmp);
    if(threadIdx.x == 0) {
        s_sum = sum_val + 1e-6f;
    }
    __syncthreads();
    if(threadIdx.x < klen) {
        ac[threadIdx.x + input_offset] = (T)(tmp / s_sum);
    }
}
#else
template<typename T>
__global__ void attn_prob_softmax_kernel(T* ac, 
                                         T* bd,
                                         const T* attn_mask, 
                                         const int sqlen, 
                                         const int klen, 
                                         const float scaler) {
    int input_offset = blockIdx.x * klen;
    int mask_offset = blockIdx.x % sqlen * klen;

    __shared__ float s_sum, s_max;

    float ac_val = threadIdx.x < klen  ? (float)ac[threadIdx.x + input_offset] : 0.0f;
    float bd_val = threadIdx.x < klen  ? (float)bd[threadIdx.x + input_offset] : 0.0f;
    float mask_val = threadIdx.x < klen ? (float)attn_mask[threadIdx.x + mask_offset] : 0.0f;

    float tmp = threadIdx.x < klen ? (ac_val + bd_val) * scaler * (1 - mask_val) - 1e20f * mask_val : -1e20f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0) {
        s_max = max_val;
    } 
    __syncthreads();

    float qk_tmp = threadIdx.x < klen ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);
    if(threadIdx.x == 0) {
        s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < klen) {
        ac[threadIdx.x + input_offset] = (T)(qk_tmp / s_sum);
    }
}

#endif

template <typename T>
void attn_prob_softmax(DType<T>* ac, 
                       DType<T>* bd, 
                       const DType<T>* attn_mask, 
                       int mlen, int sqlen, int bsz, int n_head, int d_head,
                       cudaStream_t stream) {
    float scaler = 1 / sqrt(d_head);
    dim3 grid;
    grid.x = bsz * n_head * sqlen;
    dim3 block;
    int klen = (sqlen + mlen);
    if(klen <= 32) {
        block.x = 32;
    } else if(klen > 32 && klen <= 64) {
        block.x = 64;
    } else if(klen > 64 && klen <= 128) {
        block.x = 128;
    } else if(klen > 128 && klen <= 256) {
        block.x = 256;
    } else if(klen > 256 && klen <= 512) {
        block.x = 512;
    } else { 
        block.x = 1024;
    }
    attn_prob_softmax_kernel<<<grid, block, 0, stream>>>(ac, bd, attn_mask, sqlen, klen, scaler);
}

template void attn_prob_softmax<float>(DType<float>* ac,
                                       DType<float>* bd, 
                                       const DType<float>* attn_mask, 
                                       int mlen, int sqlen, int bsz, int n_head, int d_head,
                                       cudaStream_t stream);

} /* namespace tensorflow */

