#ifndef _DELTA_INFER_CUSTOM_OPS_ALLOC_H_
#define _DELTA_INFER_CUSTOM_OPS_ALLOC_H_

#include <cstring>
#include "delta_infer/custom_ops/ops_utils.h"

namespace tensorflow {

template<typename Device, typename T>
class DeltaAlloc {
public:    
    T* malloc(size_t size) const { return nullptr; }
    void free(T* ptr) const {}
};

#ifdef WITH_CUDA
template<typename T>
class DeltaAlloc<GPUDevice, T> {
public:
    T* malloc(size_t size, OpKernelContext *context=nullptr) const {
        T* ptr = nullptr;
        //int curr_device = 0;
        //cuda(GetDevice(&curr_device));
        //printf("gpu alloc size %d (%d)\n", size, sizeof(T));
        cuda(Malloc(&ptr, size * sizeof(T)));
        return ptr;
    }

    void free(T* ptr) const {
        cuda(Free(ptr));
    }
};
#endif

template<typename T>
class DeltaAlloc<CPUDevice, T> {
public:
    DeltaAlloc() {}

    T* malloc(size_t size, OpKernelContext *context=nullptr) const {
        Tensor buf;
        long long int buf_size = (long long int)size * sizeof(T);
        tensorflow::Status status = context->allocate_temp(DT_UINT8, TensorShape{buf_size}, &buf); 
        if(status != tensorflow::Status::OK())
            throw std::runtime_error("TF error: context->allocate_temp failed"); 
                       
        auto flat = buf.flat<T>();
        T* ptr = (T*)flat.data();
        return ptr;
    }

    void free(T* ptr) const {
        printf("[DeltaFree] call from DeltaAlloc<CPUDevice, T> free\n");
    }
};

template<typename Device, typename T>
class TransformerParam;

template<typename Device, typename T>
class TransformerNLPParam;

template<typename Device, typename T, typename D = void>
struct CopyToHost {
    void operator()(T* dst, const T* src, size_t size) {}
};

template<typename Device, typename T, typename D = void>
struct CopyToDevice {
    void operator()(T* dst, const T* src, size_t size) {}
};

#ifdef WITH_CUDA
template<typename T, typename D>
struct CopyToHost<GPUDevice, T, D> {
    void operator()(T* dst, const T* src, size_t size) {
        cuda(Memcpy(dst, src, size*sizeof(T), cudaMemcpyDeviceToHost));
    }

    void async(T* dst, const T* src, size_t size, TransformerParam<GPUDevice, D>& param) {
        cuda(MemcpyAsync(dst, src, size*sizeof(T), cudaMemcpyDeviceToHost, param.stream));
    }

    void async(T* dst, const T* src, size_t size, TransformerNLPParam<GPUDevice, D>& param) {
        cuda(MemcpyAsync(dst, src, size*sizeof(T), cudaMemcpyDeviceToHost, param.stream));
    }

    void async(T* dst, const T* src, size_t size, TransformerNLPParam<CPUDevice, D>& param) {
    }
};

template<typename T, typename D>
struct CopyToDevice<GPUDevice, T, D> {
    void operator()(T* dst, const T* src, size_t size) {
        cuda(Memcpy(dst, src, size*sizeof(T), cudaMemcpyDeviceToDevice));
    }
    void async(T* dst, const T* src, size_t size, TransformerNLPParam<GPUDevice, D>& param) {
        cuda(MemcpyAsync(dst, src, size*sizeof(T), cudaMemcpyDeviceToDevice, param.stream));
    }
};
#endif

template<typename T, typename D>
struct CopyToHost<CPUDevice, T, D> {
    void operator()(T* dst, const T* src, size_t size) {
        std::memcpy(dst, src, size * sizeof(T));
    }

    void async(T* dst, const T* src, size_t size, TransformerParam<CPUDevice, D>& param) {
        std::memcpy(dst, src, size * sizeof(T));
    }

};

template<typename Device, typename T>
double CheckSum(const T* data, size_t len = 1) {
    cuda(DeviceSynchronize());
    double sum =0.0f;
    T* sum_buf =new T[len];
    CopyToHost<Device, T> cpy;
    cpy(sum_buf, data, len);
    for(int i=0; i<len; i++) {
        sum += sum_buf[i];
    }
    return sum;
}

template<typename Device, typename T>
void CheckElts(const T* data, const char* preffix, size_t len = 12) {
    cuda(DeviceSynchronize());
    T* sum_buf =new T[len];
    CopyToHost<Device, T> cpy;
    cpy(sum_buf, data, len);
    fprintf(stdout, "Check (%d)Elt of %s :", len, preffix);
    for(int i=0; i<len; i++) {
        fprintf(stdout, "%f ", sum_buf[i]);
    }
    fprintf(stdout, "\n");
    fflush(stdout); 
}

} /* namespace tensorflow */

#endif
