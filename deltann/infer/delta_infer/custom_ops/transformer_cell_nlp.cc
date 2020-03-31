#include "delta_infer/custom_ops/transformer_cell_nlp_functor.h"
#include <fstream>

namespace tensorflow {

REGISTER_OP("TransformerCellNLP")
    .Input("transformer_layer_0_rel_attn_einsum_transpose_perm: int32")
    .Input("transformer_layer_0_rel_attn_einsum_reshape_shape_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_reshape_shape_3: int32")
    .Input("transformer_layer_0_rel_attn_einsum_transpose_1_perm: int32")
    .Input("transformer_layer_0_rel_attn_einsum_reshape_1_shape_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_reshape_1_shape_2: int32")
    .Input("transformer_layer_0_rel_attn_einsum_reshape_2_shape_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_transpose_2_perm: int32")
    .Input("transformer_r_w_bias_read: T")
    .Input("transformer_layer_0_rel_attn_einsum_strided_slice_stack: int32")
    .Input("transformer_layer_0_rel_attn_einsum_strided_slice_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_strided_slice_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_einsum_mul_x: int32")
    .Input("transformer_layer_0_rel_attn_einsum_strided_slice_2_stack: int32")
    .Input("transformer_layer_0_rel_attn_einsum_strided_slice_2_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_strided_slice_2_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_einsum_mul_1_x: int32")
    .Input("transformer_layer_0_rel_attn_einsum_strided_slice_1_stack: int32")
    .Input("transformer_layer_0_rel_attn_einsum_strided_slice_1_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_strided_slice_1_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_einsum_strided_slice_3_stack: int32")
    .Input("transformer_layer_0_rel_attn_einsum_strided_slice_3_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_strided_slice_3_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_split_split_dim: int32")
    .Input("transformer_layer_0_rel_attn_reshape_1_shape_2: int32")
    .Input("transformer_layer_0_rel_attn_reshape_1_shape_3: int32")
    .Input("transformer_layer_0_rel_attn_mul_y: float")
    .Input("transformer_layer_0_rel_attn_strided_slice_3_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_3_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_reshape_shape_2: int32")
    .Input("transformer_layer_0_rel_attn_reshape_shape_3: int32")
    .Input("transformer_r_r_bias_read: T")
    .Input("transformer_layer_0_rel_attn_strided_slice_2_stack: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_2_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_2_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_4_stack: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_4_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_4_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_slice_begin: int32")
    .Input("transformer_layer_0_rel_attn_slice_size: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_stack: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_transpose_perm: int32")
    .Input("transformer_layer_0_rel_attn_qkv_tensordot_const_2: int32")
    .Input("transformer_layer_0_rel_attn_qkv_tensordot_concat_1_axis: int32")
    .Input("transformer_layer_0_rel_attn_reshape_2_shape_2: int32")
    .Input("transformer_layer_0_rel_attn_reshape_2_shape_3: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_transpose_1_perm: int32")
    .Input("transformer_dropout_identity: T")
    .Input("transformer_layer_0_rel_attn_einsum_1_transpose_2_perm: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_5_stack: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_5_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_5_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_6_stack: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_6_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_6_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_7_stack: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_7_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_7_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_8_stack: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_8_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_8_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_sub_x: float")
    .Input("transformer_layer_0_rel_attn_qkv_tensordot_reshape_1_shape: int32")
    .Input("transformer_layer_0_rel_attn_qkv_tensordot_free: int32")
    .Input("transformer_layer_0_rel_attn_qkv_tensordot_gatherv2_axis: int32")
    .Input("transformer_layer_0_rel_attn_pad_paddings: int32")
    .Input("transformer_layer_0_rel_attn_add_2_y: int32")
    .Input("transformer_concat_1: T")
    .Input("transformer_layer_0_rel_attn_strided_slice_9_stack: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_9_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_9_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_mul_2_x: float")
    .Input("transformer_layer_0_rel_attn_einsum_1_strided_slice_stack: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_strided_slice_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_strided_slice_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_strided_slice_1_stack: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_strided_slice_1_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_strided_slice_1_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_reshape_shape_0: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_reshape_shape_2: int32")
    .Input("transformer_layer_0_rel_attn_qkv_tensordot_transpose_1_perm: int32")
    .Input("transformer_layer_0_rel_attn_qkv_tensordot_const: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_strided_slice_3_stack: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_strided_slice_3_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_strided_slice_3_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_reshape_1_shape_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_reshape_1_shape_3: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_reshape_2_shape_0: int32")
    .Input("transformer_layer_0_rel_attn_concat_1_values_3: int32")
    .Input("transformer_layer_0_rel_attn_concat_1_axis: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_mul_x: int32")
    .Input("transformer_layer_0_rel_attn_qkv_tensordot_const_1: int32")
    .Input("transformer_layer_0_rel_attn_qkv_tensordot_axes: int32")
    .Input("transformer_layer_0_rel_attn_qkv_tensordot_concat_axis: int32")
    .Input("transformer_layer_0_rel_attn_qkv_kernel: float")
    .Input("transformer_layer_0_rel_attn_qkv_tensordot_gatherv2_1_axis: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_mul_1_x: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_strided_slice_2_stack: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_strided_slice_2_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_strided_slice_2_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_range_start: int32")
    .Input("transformer_layer_0_rel_attn_range_limit: int32")
    .Input("transformer_layer_0_rel_attn_range_delta: int32")
    .Input("transformer_layer_0_rel_attn_range_1_start: int32")
    .Input("transformer_layer_0_rel_attn_range_1_delta: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_transpose_1_perm: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_reshape_1_shape_0: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_reshape_1_shape_1: int32")
    .Input("transformer_layer_0_rel_attn_qkv__tensordot_const_2: int32")
    .Input("transformer_layer_0_rel_attn_qkv__tensordot_concat_1_axis: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_strided_slice_2_stack: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_strided_slice_2_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_strided_slice_2_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_reshape_shape_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_transpose_perm: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_reshape_2_shape_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_reshape_2_shape_3: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_transpose_2_perm: int32")
    .Input("transformer_layer_0_rel_attn_einsum_1_mul_2_x: int32")
    .Input("transformer_layer_0_rel_attn_rank_1: int32")
    .Input("transformer_layer_0_rel_attn_sub_2_y: int32")
    .Input("transformer_layer_0_rel_attn_concat_2_values_3: int32")
    .Input("transformer_layer_0_rel_attn_concat_2_axis: int32")
    .Input("transformer_layer_0_rel_attn_qkv__tensordot_reshape_1_shape: int32")
    .Input("transformer_layer_0_rel_attn_qkv__tensordot_free: int32")
    .Input("transformer_layer_0_rel_attn_qkv__tensordot_gatherv2_axis: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_strided_slice_stack: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_strided_slice_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_strided_slice_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_mul_x: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_strided_slice_1_stack: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_strided_slice_1_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_einsum_2_strided_slice_1_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_range_2_start: int32")
    .Input("transformer_layer_0_rel_attn_range_2_limit: int32")
    .Input("transformer_layer_0_rel_attn_range_2_delta: int32")
    .Input("transformer_layer_0_rel_attn_range_3_start: int32")
    .Input("transformer_layer_0_rel_attn_range_3_delta: int32")
    .Input("transformer_layer_0_rel_attn_reshape_3_shape_1: int32")
    .Input("transformer_layer_0_rel_attn_reshape_3_shape_2: int32")
    .Input("transformer_layer_0_rel_attn_qkv__tensordot_transpose_1_perm: int32")
    .Input("transformer_layer_0_rel_attn_qkv__tensordot_c1onst: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_10_stack: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_10_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_10_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_11_stack: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_11_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_11_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_reshape_6_shape_2: int32")
    .Input("transformer_layer_0_rel_attn_sub_3_y: int32")
    .Input("transformer_layer_0_rel_attn_r_tensordot_const_2: int32")
    .Input("transformer_layer_0_rel_attn_r_tensordot_concat_1_axis: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_1_stack: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_1_stack_1: int32")
    .Input("transformer_layer_0_rel_attn_strided_slice_1_stack_2: int32")
    .Input("transformer_layer_0_rel_attn_qkv__tensordot_const_1: int32")
    .Input("input0: T")
    .Input("transformer_layer_0_rel_attn_concat_axis: int32")
    .Input("transformer_layer_0_rel_attn_qkv__tensordot_axes: int32")
    .Input("transformer_layer_0_rel_attn_qkv__tensordot_concat_axis: int32")
    .Input("transformer_layer_0_rel_attn_qkv__kernel: float")
    .Input("transformer_layer_0_rel_attn_qkv__tensordot_gatherv2_1_axis: int32")
    .Input("transformer_layer_0_rel_attn_o_tensordot_free: int32")
    .Input("transformer_layer_0_rel_attn_o_tensordot_gatherv2_axis: int32")
    .Input("transformer_layer_0_rel_attn_o_tensordot_axes: int32")
    .Input("transformer_layer_0_rel_attn_o_tensordot_gatherv2_1_axis: int32")
    .Input("transformer_layer_0_rel_attn_o_tensordot_concat_axis: int32")
    .Input("transformer_layer_0_rel_attn_r_tensordot_reshape_1_shape: int32")
    .Input("transformer_layer_0_rel_attn_r_tensordot_free: int32")
    .Input("transformer_layer_0_rel_attn_r_tensordot_gatherv2_axis: int32")
    .Input("transformer_dropout_1_identity: T")
    .Input("transformer_layer_0_rel_attn_o_tensordot_const: int32")
    .Input("transformer_layer_0_rel_attn_o_tensordot_const_2: int32")
    .Input("transformer_layer_0_rel_attn_o_tensordot_concat_1_axis: int32")
    .Input("transformer_layer_0_rel_attn_o_tensordot_const_1: int32")
    .Input("transformer_layer_0_rel_attn_r_tensordot_transpose_1_perm: int32")
    .Input("transformer_layer_0_rel_attn_r_tensordot_const: int32")
    .Input("transformer_layer_0_rel_attn_o_tensordot_reshape_1_shape: int32")
    .Input("transformer_layer_0_rel_attn_r_tensordot_const_1: int32")
    .Input("transformer_layer_0_rel_attn_r_tensordot_axes: int32")
    .Input("transformer_layer_0_rel_attn_r_tensordot_concat_axis: int32")
    .Input("transformer_layer_0_rel_attn_r_kernel: float")
    .Input("transformer_layer_0_rel_attn_r_tensordot_gatherv2_1_axis: int32")
    .Input("transformer_layer_0_rel_attn_o_tensordot_transpose_1_perm: int32")
    .Input("transformer_layer_0_rel_attn_o_kernel: float")
    .Input("transformer_layer_0_rel_attn_layer_norm_moments_mean_reduction_indices: int32")
    .Input("transformer_layer_0_rel_attn_layer_norm_moments_variance_reduction_indices: int32")
    .Input("transformer_layer_0_rel_attn_layer_norm_batchnorm_add_y: float")
    .Input("transformer_layer_0_rel_attn_layer_norm_gamma: float")
    .Input("transformer_layer_0_rel_attn_layer_norm_beta: float")
    .Input("transformer_layer_0_ff_layer_1_tensordot_free: int32")
    .Input("transformer_layer_0_ff_layer_1_tensordot_gatherv2_axis: int32")
    .Input("transformer_layer_0_ff_layer_1_tensordot_axes: int32")
    .Input("transformer_layer_0_ff_layer_1_tensordot_gatherv2_1_axis: int32")
    .Input("transformer_layer_0_ff_layer_1_tensordot_concat_axis: int32")
    .Input("transformer_layer_0_ff_layernorm_moments_mean_reduction_indices: int32")
    .Input("transformer_layer_0_ff_layer_1_tensordot_const: int32")
    .Input("transformer_layer_0_ff_layer_1_tensordot_const_2: int32")
    .Input("transformer_layer_0_ff_layer_1_tensordot_concat_1_axis: int32")
    .Input("transformer_layer_0_ff_layer_1_tensordot_const_1: int32")
    .Input("transformer_layer_0_ff_layernorm_moments_variance_reduction_indices: int32")
    .Input("transformer_layer_0_ff_layer_1_tensordot_reshape_1_shape: int32")
    .Input("transformer_layer_0_ff_layer_2_bias: float")
    .Input("transformer_layer_0_ff_layernorm_batchnorm_add_y: float")
    .Input("transformer_layer_0_ff_layernorm_gamma: float")
    .Input("transformer_layer_0_ff_layer_1_tensordot_transpose_1_perm: int32")
    .Input("transformer_layer_0_ff_layer_2_tensordot_const_2: int32")
    .Input("transformer_layer_0_ff_layer_2_tensordot_concat_1_axis: int32")
    .Input("transformer_layer_0_ff_layernorm_beta: float")
    .Input("transformer_layer_0_ff_layer_1_bias: float")
    .Input("transformer_layer_0_ff_layer_1_kernel: float")
    .Input("transformer_layer_0_ff_layer_2_tensordot_reshape_1_shape: int32")
    .Input("transformer_layer_0_ff_layer_2_tensordot_free: int32")
    .Input("transformer_layer_0_ff_layer_2_tensordot_gatherv2_axis: int32")
    .Input("transformer_layer_0_ff_layer_2_tensordot_transpose_1_perm: int32")
    .Input("transformer_layer_0_ff_layer_2_tensordot_const: int32")
    .Input("transformer_layer_0_ff_layer_2_tensordot_const_1: int32")
    .Input("transformer_layer_0_ff_layer_2_tensordot_axes: int32")
    .Input("transformer_layer_0_ff_layer_2_tensordot_concat_axis: int32")
    .Input("transformer_layer_0_ff_layer_2_kernel: float")
    .Input("transformer_layer_0_ff_layer_2_tensordot_gatherv2_1_axis: int32")
    .Input("transformer_layer_0_ff_layer_2_1_tensordot_free: int32")
    .Input("transformer_layer_0_ff_layer_2_1_tensordot_gatherv2_axis: int32")
    .Input("transformer_layer_0_ff_layer_2_1_tensordot_axes: int32")
    .Input("transformer_layer_0_ff_layer_2_1_tensordot_gatherv2_1_axis: int32")
    .Input("transformer_layer_0_ff_layer_2_1_tensordot_concat_axis: int32")
    .Input("transformer_layer_0_ff_layer_2_1_bias: float")
    .Input("transformer_layer_0_ff_layer_2_1_tensordot_const: int32")
    .Input("transformer_layer_0_ff_layer_2_1_tensordot_const_2: int32")
    .Input("transformer_layer_0_ff_layer_2_1_tensordot_concat_1_axis: int32")
    .Input("transformer_layer_0_ff_layer_2_1_tensordot_const_1: int32")
    .Input("transformer_layer_0_ff_layer_2_1_tensordot_reshape_1_shape: int32")
    .Input("transformer_layer_0_ff_layer_2_1_tensordot_transpose_1_perm: int32")
    .Input("transformer_layer_0_ff_layer_2_1_kernel: float")
    .Output("output: T")
    .Attr("T: {float, half}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(49));
        return Status::OK();
    });

template<typename Device, typename T>
class TansformerCellNLPOp : public OpKernel {
    typedef DeltaTraits<Device, T> traits;
public:
    explicit TansformerCellNLPOp(OpKernelConstruction* context) : OpKernel(context) {
    }

    ~TansformerCellNLPOp() {
    }

    void Compute(OpKernelContext* context) override {
        // initial paramter
        _param.init(context);
        auto r_w_bias_shape  = context->input(8).shape();
        auto w_shape = context->input(49).shape();
        auto mems_shape = context->input(158).shape();
        _param.n_head = r_w_bias_shape.dim_size(0);
        _param.d_head = r_w_bias_shape.dim_size(1);
        _param.batch_size = w_shape.dim_size(1);
        _param.sqlen = w_shape.dim_size(0);
        _param.d_model = w_shape.dim_size(2);
        _param.mlen = mems_shape.dim_size(0);
        //printf("%d, %d, %d, %d, %d, %d\n", _param.n_head, _param.d_head, _param.batch_size, _param.sqlen, _param.d_model, _param.mlen);
        auto kernel1_shape = context->input(212).shape();
        auto kernel2_shape = context->input(221).shape();
        _param.kernel1_dim = kernel1_shape.dim_size(1);
        _param.kernel2_dim = kernel2_shape.dim_size(0);
                
        DELTA_SCOPE{
            auto test_shape = context->input(8).shape();
            printf("Get r_w_bias inputshape[%d]: [%d, %d]\n", test_shape.dims(), test_shape.dim_size(0), test_shape.dim_size(1)); 
            test_shape = context->input(31).shape();
            printf("Get r_r_bias inputshape[%d]: [%d, %d]\n", test_shape.dims(), test_shape.dim_size(0), test_shape.dim_size(1)); 
            test_shape = context->input(49).shape();
            printf("Get w inputshape[%d]: [%d, %d, %d]\n", test_shape.dims(), test_shape.dim_size(0), test_shape.dim_size(1), test_shape.dim_size(2)); 
            test_shape = context->input(69).shape();
            printf("Get attn_mask inputshape[%d]: [%d, %d]\n", test_shape.dims(), test_shape.dim_size(0), test_shape.dim_size(1));
            test_shape = context->input(158).shape();
            printf("Get mems inputshape[%d]: [%d, %d, %d]\n", test_shape.dims(), test_shape.dim_size(0), test_shape.dim_size(1), test_shape.dim_size(2));
            test_shape = context->input(172).shape();
            printf("Get pos_emb inputshape[%d]: [%d, %d, %d]\n", test_shape.dims(), test_shape.dim_size(0), test_shape.dim_size(1), test_shape.dim_size(2));
            test_shape = context->input(212).shape();
            printf("Get ff dense kernel1 inputshape[%d]: [%d, %d, %d]\n", test_shape.dims(), test_shape.dim_size(0), test_shape.dim_size(1));
            test_shape = context->input(221).shape();
            printf("Get ff dense kernel2 inputshape[%d]: [%d, %d, %d]\n", test_shape.dims(), test_shape.dim_size(0), test_shape.dim_size(1));
            test_shape = context->input(235).shape();
            printf("Get ff dense kernel3 inputshape[%d]: [%d, %d, %d]\n", test_shape.dims(), test_shape.dim_size(0), test_shape.dim_size(1));
        };

        _param.r_w_bias = reinterpret_cast<const typename traits::DataType *>(context->input(8).flat<T>().data());
        _param.r_r_bias = reinterpret_cast<const typename traits::DataType *>(context->input(31).flat<T>().data());
        _param.w = reinterpret_cast<const typename traits::DataType *>(context->input(49).flat<T>().data());
        _param.attn_mask = reinterpret_cast<const typename traits::DataType *>(context->input(69).flat<T>().data());
        _param.mems = reinterpret_cast<const typename traits::DataType *>(context->input(158).flat<T>().data());
        _param.pos_emb = reinterpret_cast<const typename traits::DataType *>(context->input(172).flat<T>().data());
        _param.att_kernel_QKV = reinterpret_cast<const typename traits::DataType *>(context->input(96).flat<T>().data());
        _param.att_kernel_QKV_1 = reinterpret_cast<const typename traits::DataType *>(context->input(162).flat<T>().data());
        /*DELTA_SCOPE{
            CopyToHost<GPUDevice, float, T> copy_to_host;
            float * tmp_w = new float[256 * 225];
            copy_to_host.async(tmp_w, _param.att_kernel_QKV_1, 256*225, _param);
            //CheckElts<GPUDevice, float>(tmp_w, "tmp_w"); 
            double sum = 0.0;
            for(int i = 0; i < 256; i++) {
                printf("get w data(%d): %f\n", i, tmp_w[i*225 + 0]);
                sum += tmp_w[i*225 + 0];
            }
            printf("didi sum of tmp_w first col: %f\n", sum);
        };*/
        _param.att_kernel_r = reinterpret_cast<const typename traits::DataType *>(context->input(183).flat<T>().data());
        _param.att_kernel_o = reinterpret_cast<const typename traits::DataType *>(context->input(186).flat<T>().data());
        _param.att_layernorm_gamma = reinterpret_cast<const typename traits::DataType *>(context->input(190).flat<T>().data());
        _param.att_layernorm_beta = reinterpret_cast<const typename traits::DataType *>(context->input(191).flat<T>().data());
        _param.ff_layernorm_gamma = reinterpret_cast<const typename traits::DataType *>(context->input(206).flat<T>().data());
        _param.ff_layernorm_beta = reinterpret_cast<const typename traits::DataType *>(context->input(210).flat<T>().data());
        _param.ff_layer_1_kernel = reinterpret_cast<const typename traits::DataType *>(context->input(212).flat<T>().data());
        _param.ff_layer_1_bias = reinterpret_cast<const typename traits::DataType *>(context->input(211).flat<T>().data());
        _param.ff_layer_2_kernel = reinterpret_cast<const typename traits::DataType *>(context->input(221).flat<T>().data());
        _param.ff_layer_2_bias = reinterpret_cast<const typename traits::DataType *>(context->input(204).flat<T>().data());
        _param.ff_layer_2_1_kernel = reinterpret_cast<const typename traits::DataType *>(context->input(235).flat<T>().data());
        _param.ff_layer_2_1_bias = reinterpret_cast<const typename traits::DataType *>(context->input(228).flat<T>().data());
        
        OP_REQUIRES(context, _param.r_w_bias != nullptr, errors::InvalidArgument("r_w_bias is null"));
        OP_REQUIRES(context, _param.r_r_bias != nullptr, errors::InvalidArgument("r_r_bias is null"));
        OP_REQUIRES(context, _param.w != nullptr, errors::InvalidArgument("w is null"));
        OP_REQUIRES(context, _param.attn_mask != nullptr, errors::InvalidArgument("attn_mask is null"));
        OP_REQUIRES(context, _param.mems != nullptr, errors::InvalidArgument("mems is null"));
        OP_REQUIRES(context, _param.pos_emb != nullptr, errors::InvalidArgument("pos_emb` is null"));
        OP_REQUIRES(context, _param.att_kernel_QKV != nullptr, errors::InvalidArgument("att_kernel_QKV is null"));
        OP_REQUIRES(context, _param.att_kernel_QKV_1 != nullptr, errors::InvalidArgument("att_kernel_QKV_1 is null"));
        OP_REQUIRES(context, _param.att_kernel_r != nullptr, errors::InvalidArgument("att_kernel_r is null"));
        OP_REQUIRES(context, _param.att_kernel_o != nullptr, errors::InvalidArgument("att_kernel_o is null"));
        OP_REQUIRES(context, _param.att_layernorm_gamma != nullptr, errors::InvalidArgument("att_layernorm_gamma is null"));
        OP_REQUIRES(context, _param.att_layernorm_beta != nullptr, errors::InvalidArgument("att_layernorm_beta is null"));
        OP_REQUIRES(context, _param.ff_layernorm_gamma != nullptr, errors::InvalidArgument("ff_layernorm_gamma is null"));
        OP_REQUIRES(context, _param.ff_layernorm_beta != nullptr, errors::InvalidArgument("ff_layernorm_beta is null"));
        OP_REQUIRES(context, _param.ff_layer_1_kernel != nullptr, errors::InvalidArgument("ff_layer_1_kernel is null"));
        OP_REQUIRES(context, _param.ff_layer_1_bias != nullptr, errors::InvalidArgument("ff_layer_1_bias is null"));
        OP_REQUIRES(context, _param.ff_layer_2_kernel != nullptr, errors::InvalidArgument("ff_layer_2_kernel is null"));
        OP_REQUIRES(context, _param.ff_layer_2_bias != nullptr, errors::InvalidArgument("ff_layer_2_bias is null"));
        OP_REQUIRES(context, _param.ff_layer_2_1_kernel != nullptr, errors::InvalidArgument("ff_layer_2_1_kernel is null"));
        OP_REQUIRES(context, _param.ff_layer_2_1_bias != nullptr, errors::InvalidArgument("ff_layer_2_1_bias is null"));

        Tensor *output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, {w_shape.dim_size(0), w_shape.dim_size(1), w_shape.dim_size(2)}, &output));
        _param.transformer_out = reinterpret_cast<typename traits::DataType *>(output->flat<T>().data());
        // initial
        _functor.init(_param);
        // forward
        _functor(context, _param);
    }

private:
    CopyToHost<Device, float, T> _copy_to_host;
    TransformerNLPParam<Device, T>  _param;
    TransformerCellNLPFunctor<Device, T> _functor;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TransformerCellNLP").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TansformerCellNLPOp<CPUDevice, T>);

REGISTER_CPU(float);

// Register the GPU kernels.
#ifdef WITH_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TransformerCellNLP").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      TansformerCellNLPOp<GPUDevice, T>);

REGISTER_GPU(float);

#endif  // GOOGLE_CUDA


} /* namespace tensorflow */
