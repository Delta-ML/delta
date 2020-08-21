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

#include "../../../kernels/zcr.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace delta {
class ZcrOp : public OpKernel {
 public:
  explicit ZcrOp(OpKernelConstruction* context) : OpKernel(context) {
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
    ZCR cls_zcr;
    cls_zcr.set_window_length_sec(window_length_);
    cls_zcr.set_frame_length_sec(frame_length_);
    OP_REQUIRES(
        context, cls_zcr.init_zcr(L, sample_rate),
        errors::InvalidArgument("zcr_class initialization failed for length ",
                                L, " and sample rate ", sample_rate));

    Tensor* output_tensor = nullptr;
    int i_WinLen = static_cast<int>(window_length_ * sample_rate);
    int i_FrmLen = static_cast<int>(frame_length_ * sample_rate);
    int i_NumFrm = (L - i_WinLen) / i_FrmLen + 1;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({1, i_NumFrm}), &output_tensor));

    const float* input_flat = input_tensor.flat<float>().data();
    float* output_flat = output_tensor->flat<float>().data();

    int ret;
    ret = cls_zcr.proc_zcr(input_flat);
    ret = cls_zcr.get_zcr(output_flat);
  }

 private:
  float window_length_;
  float frame_length_;
};

REGISTER_KERNEL_BUILDER(Name("ZCR").Device(DEVICE_CPU), ZcrOp);

}  // namespace delta
