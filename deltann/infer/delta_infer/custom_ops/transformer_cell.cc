#include "delta_infer/custom_ops/transformer_cell_functor.h"
#include <fstream>

namespace tensorflow {

REGISTER_OP("TransformerCell")
    .Input("self_transpose_perm: int32") 
    .Input("self_transpose_1_perm: int32")
    .Input("layer_0_attention_self_mul_y: float")
    .Input("layer_0_attention_self_reshape_shape: int32")
    .Input("layer_0_attention_self_reshape_1_shape: int32")
    .Input("layer_0_attention_self_mul_1_y: float")
    .Input("layer_0_attention_self_query_bias: float")
    .Input("reshape: T")
    .Input("layer_0_attention_self_key_bias: float")
    .Input("layer_0_attention_self_sub_x: float")
    .Input("layer_0_attention_self_query_kernel: float")
    .Input("layer_0_attention_self_key_kernel: float")
    .Input("layer_0_attention_self_transpose_2_perm: int32")
    .Input("layer_0_attention_self_transpose_3_perm: int32")
    .Input("place_holder_1: int32")
    .Input("layer_0_attention_self_expanddims_dim: int32")
    .Input("layer_0_attention_self_reshape_2_shape: int32")
    .Input("layer_0_attention_self_reshape_3_shape: int32")
    .Input("layer_0_attention_self_value_bias: float")
    .Input("layer_0_attention_output_dense_kernel: float")
    .Input("layer_0_attention_self_value_kernel: float")
    .Input("layer_0_attention_output_dense_bias: float")
    .Input("layer_0_attention_output_layernorm_moments_mean_reduction_indices: int32")
    .Input("layer_0_attention_output_layernorm_moments_variance_reduction_indices: int32")
    .Input("layer_0_attention_output_layernorm_batchnorm_add_y: float")
    .Input("layer_0_attention_output_layernorm_gamma: float")
    .Input("layer_0_attention_output_layernorm_beta: float")
    .Input("layer_0_intermediate_dense_kernel: float")
    .Input("layer_0_output_layernorm_moments_mean_reduction_indices: int32")
    .Input("layer_0_intermediate_dense_bias: float")
    .Input("layer_0_intermediate_dense_pow_y: float")
    .Input("layer_0_output_dense_bias: float")
    .Input("layer_0_output_layernorm_moments_variance_reduction_indices: int32")
    .Input("layer_0_intermediate_dense_mul_x: float")
    .Input("layer_0_intermediate_dense_mul_1_x: float")
    .Input("layer_0_intermediate_dense_mul_2_x: float")
    .Input("layer_0_output_dense_kernel: float")
    .Input("layer_0_output_layernorm_batchnorm_add_y: float")
    .Input("layer_0_output_layernorm_gamma: float")
    .Input("layer_0_intermediate_dense_add_1_x: float")
    .Input("layer_0_output_layernorm_beta: float")
    .Output("output: T")
    .Attr("T: {float, half}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        /*auto shape_hdl  = c->input(4);
        auto batch_size = c->Dim(shape_hdl, 0);
        auto from_seq_len = c->Dim(shape_hdl, 1);
        //auto to_seq_len = c->Dim(shape_hdl, 1);
        auto head_num = c->Dim(shape_hdl, 2);
        auto size_per_head = c->Dim(shape_hdl, 3);
        shape_inference::DimensionHandle out1;
        c->Multiply(batch_size, from_seq_len, &out1);
        shape_inference::DimensionHandle out2;
        c->Multiply(head_num, size_per_head, &out2);
        c->set_output(0, c->MakeShape({out1, out2}));*/
        c->set_output(0, c->input(7));
        return Status::OK();
    });

std::once_flag flag;

void NEWPrintTensor(float* data, const char* file_path) {
    std::call_once(flag, [&]() {
        printf("get into temp of transformer output tensor\n");
        std::ofstream out(file_path, std::ios_base::binary);
        if(out.good()) {
            for(int i=0; i<100; i++) {
                //std::cout << data[i] << "\n";
                //out.write((char *)&(data[i]), sizeof(float));
                out << data[i] << "\n";
            }
        }
        out.close();
    });
}

template<typename Device, typename T>
class TansformerCellOp : public OpKernel {
    typedef DeltaTraits<Device, T> traits;
public:
    explicit TansformerCellOp(OpKernelConstruction* context) : OpKernel(context) {
        host_shape = new int[4]; 
    }

    ~TansformerCellOp() {
        if(!host_shape) {
            delete host_shape;
            host_shape = nullptr;
        }
    }

    void Compute(OpKernelContext* context) override {
        // initial paramter
        _param.init(context);
#if 0
        const int* shape_val = reinterpret_cast<const int*>(context->input(3).flat<int32>().data());
        if(_param.batch_size < 0) {
            _copy_to_host(host_shape, shape_val, 4);
            _param.batch_size = host_shape[0];
            _param.from_seq_len = host_shape[1];
            _param.to_seq_len = host_shape[1];
            _param.head_num = host_shape[2];
            _param.size_per_head = host_shape[3];
        }
#else
        auto real_time_shape = context->input(7).shape();
        int len = real_time_shape.dims();
        if(len == 2) {
            const int* shape_val = reinterpret_cast<const int*>(context->input(3).flat<int32>().data());
            _copy_to_host.async(host_shape, shape_val, 4, _param);
            _param.batch_size = host_shape[0];
            _param.from_seq_len = host_shape[1];
            _param.to_seq_len = host_shape[1];
            _param.head_num = host_shape[2];
            _param.size_per_head = host_shape[3];
            /*_param.from_seq_len = context->input(17).shape().dim_size(1);
            _param.to_seq_len = context->input(17).shape().dim_size(1);
            int fused_seq_batch = real_time_shape.dim_size(0);
            _param.batch_size = fused_seq_batch / _param.from_seq_len;*/
        } else if(len == 4) {
            _param.batch_size = real_time_shape.dim_size(0);
            _param.from_seq_len = real_time_shape.dim_size(1);
            _param.to_seq_len = real_time_shape.dim_size(1);
            _param.head_num = real_time_shape.dim_size(2);
            _param.size_per_head = real_time_shape.dim_size(3);
        } else {
            fprintf(stderr, "Dims of input shape should be either 2 or 4 but you got %d\n", len);
            exit(1);
        }
#endif
        //printf("Get inputshape: [%d, %d, %d, %d]\n", _param.batch_size, _param.from_seq_len, _param.head_num, _param.size_per_head);
        _param.from_tensor = reinterpret_cast<const typename traits::DataType *>(context->input(7).flat<T>().data());
        _param.to_tensor = reinterpret_cast<const typename traits::DataType *>(context->input(7).flat<T>().data());
        _param.kernel_Q = reinterpret_cast<const typename traits::DataType *>(context->input(10).flat<T>().data());
        _param.kernel_K = reinterpret_cast<const typename traits::DataType *>(context->input(11).flat<T>().data());
        _param.kernel_V = reinterpret_cast<const typename traits::DataType *>(context->input(20).flat<T>().data());
        _param.bias_Q = reinterpret_cast<const typename traits::DataType *>(context->input(6).flat<T>().data());
        _param.bias_K = reinterpret_cast<const typename traits::DataType *>(context->input(8).flat<T>().data());
        _param.bias_V = reinterpret_cast<const typename traits::DataType *>(context->input(18).flat<T>().data());
        // mask
        _param.mask = reinterpret_cast<const int *>(context->input(14).flat<int>().data());
        _param.att_output_kernel = reinterpret_cast<const typename traits::DataType *>(context->input(19).flat<T>().data());
        _param.att_output_bias = reinterpret_cast<const typename traits::DataType *>(context->input(21).flat<T>().data());
        _param.att_output_layernorm_beta = reinterpret_cast<const typename traits::DataType *>(context->input(26).flat<T>().data());
        _param.att_output_layernorm_gamma = reinterpret_cast<const typename traits::DataType *>(context->input(25).flat<T>().data());
        _param.inter_kernel = reinterpret_cast<const typename traits::DataType *>(context->input(27).flat<T>().data());
        _param.inter_bias = reinterpret_cast<const typename traits::DataType *>(context->input(29).flat<T>().data());
        _param.output_kernel = reinterpret_cast<const typename traits::DataType *>(context->input(36).flat<T>().data());
        _param.output_bias = reinterpret_cast<const typename traits::DataType *>(context->input(31).flat<T>().data());
        _param.output_layernorm_beta = reinterpret_cast<const typename traits::DataType *>(context->input(40).flat<T>().data());
        _param.output_layernorm_gamma = reinterpret_cast<const typename traits::DataType *>(context->input(38).flat<T>().data()); 

        OP_REQUIRES(context, _param.from_tensor != nullptr, errors::InvalidArgument("from tensor is null"));
        OP_REQUIRES(context, _param.to_tensor != nullptr, errors::InvalidArgument("to tensor is null"));
        OP_REQUIRES(context, _param.kernel_Q != nullptr, errors::InvalidArgument("attr_kernel_Q is null"));
        OP_REQUIRES(context, _param.kernel_K != nullptr, errors::InvalidArgument("attr_kernel_K is null"));
        OP_REQUIRES(context, _param.kernel_V != nullptr, errors::InvalidArgument("attr_kernel_V is null"));
        OP_REQUIRES(context, _param.bias_Q != nullptr, errors::InvalidArgument("attr_bias_Q is null"));
        OP_REQUIRES(context, _param.bias_K != nullptr, errors::InvalidArgument("attr_bias_K is null"));
        OP_REQUIRES(context, _param.bias_V != nullptr, errors::InvalidArgument("attr_bias_V is null"));
        OP_REQUIRES(context, _param.mask != nullptr, errors::InvalidArgument("attr_mask is null"));
        OP_REQUIRES(context, _param.att_output_kernel != nullptr, errors::InvalidArgument("attr_output_kernel is null"));
        OP_REQUIRES(context, _param.att_output_bias != nullptr, errors::InvalidArgument("attr_output_bias is null"));
        OP_REQUIRES(context, _param.att_output_layernorm_beta != nullptr, errors::InvalidArgument("attr_output_layernorm_beta is null"));
        OP_REQUIRES(context, _param.att_output_layernorm_gamma != nullptr, errors::InvalidArgument("attr_output_layernorm_gamma is null"));
        OP_REQUIRES(context, _param.inter_kernel != nullptr, errors::InvalidArgument("inter_kernel is null"));
        OP_REQUIRES(context, _param.inter_bias != nullptr, errors::InvalidArgument("inter_bias is null"));
        OP_REQUIRES(context, _param.output_kernel != nullptr, errors::InvalidArgument("output_kernel is null"));
        OP_REQUIRES(context, _param.output_bias != nullptr, errors::InvalidArgument("output_bias is null"));
        OP_REQUIRES(context, _param.output_layernorm_beta != nullptr, errors::InvalidArgument("output_layernorm_beta is null"));
        OP_REQUIRES(context, _param.output_layernorm_gamma != nullptr, errors::InvalidArgument("output_layernorm_gamma is null"));
        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, {_param.batch_size * _param.from_seq_len, _param.head_num * _param.size_per_head}, &output));
        _param.transformer_out = reinterpret_cast<typename traits::DataType *>(output->flat<T>().data());
        // initial
        _functor.init(_param);
        // forward
        _functor(context, _param);
#if 0
        DELTA_SCOPE{
            // check output
            cuda(DeviceSynchronize());
            T* out = new T[_param.batch_size * _param.from_seq_len * _param.head_num * _param.size_per_head];
            CopyToHost<Device, T> cpy;
            cpy(out, _param.transformer_out, _param.batch_size * _param.from_seq_len * _param.head_num * _param.size_per_head);
            NEWPrintTensor(out, "temp_results.txt");
        };
#endif

    }

private:
    int* host_shape{nullptr};
    CopyToHost<Device, int, T> _copy_to_host;
    T* _swap_space;
    TransformerParam<Device, T>  _param;
    TransformerCellFunctor<Device, T> _functor;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TransformerCell").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TansformerCellOp<CPUDevice, T>);

REGISTER_CPU(float);

// Register the GPU kernels.
#ifdef WITH_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TransformerCell").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      TansformerCellOp<GPUDevice, T>);

REGISTER_GPU(float);

#endif  // GOOGLE_CUDA


} /* namespace tensorflow */
