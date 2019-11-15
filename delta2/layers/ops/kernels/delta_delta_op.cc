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

// See docs in ../ops/audio_ops.cc
#include "kernels/delta_delta.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

// https://github.com/eigenteam/eigen-git-mirror/blob/master/unsupported/Eigen/CXX11/src/Tensor/README.md

namespace delta {

// add first and seoncd order derivatives
class DeltaDeltaOp : public OpKernel {
 public:
  explicit DeltaDeltaOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("order", &order_));
    OP_REQUIRES_OK(context, context->GetAttr("window", &window_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& feats = context->input(0);
    OP_REQUIRES(context, feats.dims() == 2,
                errors::InvalidArgument("features must be 2-dimensional",
                                        feats.shape().DebugString()));
    // feats shape [time, feat dim]
    const int time = feats.dim_size(0);  // num frames
    const int feat_dim = feats.dim_size(1);
    const int output_dim = feat_dim * (order_ + 1);

    DeltaDelta delta;
    OP_REQUIRES(
        context, delta.Initialize(order_, window_),
        errors::InvalidArgument("DeltaDelta initialization failed for order ",
                                order_, " and window ", window_));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({time, output_dim}),
                                            &output_tensor));

    // TType<float>::Tensor feats_t = feats.tensor<float, feats.dims()>;
    float* output_flat = output_tensor->flat<float>().data();

    for (int t = 0; t < time; t++) {
      float* row = output_flat + t * output_dim;

      // add delta-delta
      std::vector<double> out;
      delta.Compute(feats, t, &out);

      // fill output buffer
      DCHECK_EQ(output_dim, out.size());
      for (int i = 0; i < output_dim; i++) {
        row[i] = static_cast<float>(out[i]);
      }
    }
  }

 private:
  int order_;
  int window_;
};

REGISTER_KERNEL_BUILDER(Name("DeltaDelta").Device(DEVICE_CPU), DeltaDeltaOp);

}  // namespace delta
