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
#include "kernels/mfcc_dct.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace delta {

class MfccDctOp : public OpKernel {
 public:
  explicit MfccDctOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("coefficient_count",
                                             &coefficient_count_));
    OP_REQUIRES_OK(context, context->GetAttr("cepstral_lifter",
                                             &cepstral_lifter_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& fbank = context->input(0);
    OP_REQUIRES(context, fbank.dims() == 3,
                errors::InvalidArgument("Fbank must be 3-dimensional",
                                        fbank.shape().DebugString()));
    const Tensor& sample_rate_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(sample_rate_tensor.shape()),
                errors::InvalidArgument(
                    "Input sample_rate should be a scalar tensor, got ",
                    sample_rate_tensor.shape().DebugString(), " instead."));
    const int32 sample_rate = sample_rate_tensor.scalar<int32>()();

    // shape [channels, time, bins]
    const int fbank_channels = fbank.dim_size(2);
    const int fbank_samples = fbank.dim_size(1);
    const int audio_channels = fbank.dim_size(0);

    MfccDct mfcc;
    mfcc.set_coefficient_count(coefficient_count_);
    mfcc.set_cepstral_lifter(cepstral_lifter_);

    OP_REQUIRES(context, mfcc.Initialize(fbank_channels, coefficient_count_),
                errors::InvalidArgument(
                    "MFCC initialization failed for fbank channel ",
                    fbank_channels, " and  coefficient count", coefficient_count_));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0,
                       TensorShape({audio_channels, fbank_samples, coefficient_count_}),
                       &output_tensor));

    const float* fbank_flat = fbank.flat<float>().data();
    float* output_flat = output_tensor->flat<float>().data();

    for (int audio_channel = 0; audio_channel < audio_channels;
         ++audio_channel) {
      for (int fbank_sample = 0; fbank_sample < fbank_samples;
           ++fbank_sample) {
        const float* sample_data =
            fbank_flat +
            (audio_channel * fbank_samples * fbank_channels) +
            (fbank_sample * fbank_channels);
        std::vector<double> mfcc_input(sample_data,
                                        sample_data + fbank_channels);
        std::vector<double> mfcc_output;
        mfcc.Compute(mfcc_input, &mfcc_output);
        DCHECK_EQ(coefficient_count_, mfcc_output.size());
        float* output_data =
            output_flat +
            (audio_channel * fbank_samples * coefficient_count_) +
            (fbank_sample * coefficient_count_);
        for (int i = 0; i < coefficient_count_; ++i) {
          output_data[i] = mfcc_output[i];
        }
      }
    }

  }

 private:
  float cepstral_lifter_;
  int coefficient_count_;
};

REGISTER_KERNEL_BUILDER(Name("MfccDct").Device(DEVICE_CPU), MfccDctOp);

}  // namespace delta
