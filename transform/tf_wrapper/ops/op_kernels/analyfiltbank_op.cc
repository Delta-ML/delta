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

#include "../../../kernels/analyfiltbank.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace delta {
class AfbOp : public OpKernel {
 public:
  explicit AfbOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("window_length", &window_length_));
    OP_REQUIRES_OK(context, context->GetAttr("frame_length", &frame_length_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    OP_REQUIRES(context, input_tensor.dims() == 1,
                errors::InvalidArgument("input signal must be 1-dimensional",
                                        input_tensor.shape().DebugString()));

    const Tensor& sample_rate_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(sample_rate_tensor.shape()),
                errors::InvalidArgument(
                    "Input sample rate should be a scalar tensor, got ",
                    sample_rate_tensor.shape().DebugString(), " instead."));
    const float sample_rate = sample_rate_tensor.scalar<float>()();

    // shape
    const int L = input_tensor.dim_size(0);
    Analyfiltbank cls_afb;
    cls_afb.set_window_length_sec(window_length_);
    cls_afb.set_frame_length_sec(frame_length_);
    OP_REQUIRES(context, cls_afb.init_afb(L, sample_rate),
                errors::InvalidArgument(
                    "analyfiltbank_class initialization failed for length ", L,
                    " and sample rate ", sample_rate));

    Tensor* output_tensor_1 = nullptr;
    Tensor* output_tensor_2 = nullptr;
    int i_WinLen = static_cast<int>(window_length_ * sample_rate);
    int i_FrmLen = static_cast<int>(frame_length_ * sample_rate);
    int i_NumFrm = (L - i_WinLen) / i_FrmLen + 1;
    int i_FrqNum = static_cast<int>(pow(2.0f, ceil(log2(i_WinLen))) / 2 + 1);
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({i_NumFrm, i_FrqNum}),
                                          &output_tensor_1));
    OP_REQUIRES_OK(
        context, context->allocate_output(1, TensorShape({i_NumFrm, i_FrqNum}),
                                          &output_tensor_2));

    const float* input_flat = input_tensor.flat<float>().data();
    float* output_flat_1 = output_tensor_1->flat<float>().data();
    float* output_flat_2 = output_tensor_2->flat<float>().data();

    int ret;
    ret = cls_afb.proc_afb(input_flat);
    ret = cls_afb.get_afb(output_flat_1, output_flat_2);
  }

 private:
  float window_length_;
  float frame_length_;
};

REGISTER_KERNEL_BUILDER(Name("Analyfiltbank").Device(DEVICE_CPU), AfbOp);

}  // namespace delta
