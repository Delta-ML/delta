/* Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "../../../kernels/synthfiltbank.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace delta {
class SfbOp : public OpKernel {
 public:
  explicit SfbOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("window_length", &window_length_));
    OP_REQUIRES_OK(context, context->GetAttr("frame_length", &frame_length_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor_1 = context->input(0);
    OP_REQUIRES(context, input_tensor_1.dims() == 2,
                errors::InvalidArgument("input signal must be 2-dimensional",
                                        input_tensor_1.shape().DebugString()));

    const Tensor& input_tensor_2 = context->input(1);
    OP_REQUIRES(context, input_tensor_2.dims() == 2,
                errors::InvalidArgument("input signal must be 2-dimensional",
                                        input_tensor_2.shape().DebugString()));

    const Tensor& sample_rate_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(sample_rate_tensor.shape()),
                errors::InvalidArgument(
                    "Input sample rate should be a scalar tensor, got ",
                    sample_rate_tensor.shape().DebugString(), " instead."));
    const float sample_rate = sample_rate_tensor.scalar<float>()();

    // shape
    const int i_NumFrm = input_tensor_1.dim_size(0);
    const int i_NumFrq = input_tensor_1.dim_size(1);
    Synthfiltbank cls_sfb;
    cls_sfb.set_window_length_sec(window_length_);
    cls_sfb.set_frame_length_sec(frame_length_);
    OP_REQUIRES(
        context, cls_sfb.init_sfb(i_NumFrq, i_NumFrm, sample_rate),
        errors::InvalidArgument(
            "synthfiltbank_class initialization failed for Frq_Dim ", i_NumFrq,
            "Frm_Dim ", i_NumFrm, " and sample rate ", sample_rate));

    Tensor* output_tensor = nullptr;
    int i_WinLen = static_cast<int>(window_length_ * sample_rate);
    int i_FrmLen = static_cast<int>(frame_length_ * sample_rate);
    int L = (i_NumFrm - 1) * i_FrmLen + i_WinLen;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({1, L}),
                                                     &output_tensor));

    const float* input_flat_1 = input_tensor_1.flat<float>().data();
    const float* input_flat_2 = input_tensor_2.flat<float>().data();
    float* output_flat = output_tensor->flat<float>().data();

    int ret;
    ret = cls_sfb.proc_sfb(input_flat_1, input_flat_2);
    ret = cls_sfb.get_sfb(output_flat);
  }

 private:
  float window_length_;
  float frame_length_;
};

REGISTER_KERNEL_BUILDER(Name("Synthfiltbank").Device(DEVICE_CPU), SfbOp);

}  // namespace delta
