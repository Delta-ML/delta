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

#include <string.h>
#include "../../../kernels/add_rir_noise_aecres/audio.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace delta {
class AddRirNoiseAecresOp : public OpKernel {
 public:
  explicit AddRirNoiseAecresOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("snr_min", &snr_min_));
    OP_REQUIRES_OK(context, context->GetAttr("snr_max", &snr_max_));
    OP_REQUIRES_OK(context, context->GetAttr("if_add_rir", &if_add_rir_));
    OP_REQUIRES_OK(context, context->GetAttr("rir_filelist", &rir_filelist_));
    OP_REQUIRES_OK(context, context->GetAttr("if_add_noise", &if_add_noise_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("noise_filelist", &noise_filelist_));
    OP_REQUIRES_OK(context, context->GetAttr("if_add_aecres", &if_add_aecres_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("aecres_filelist", &aecres_filelist_));
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
    const int sample_rate1 = static_cast<int>(sample_rate);

    // shape
    const int L = input_tensor.dim_size(0);
    char* rir_filelist = const_cast<char*>(rir_filelist_.c_str());
    char* noise_filelist = const_cast<char*>(noise_filelist_.c_str());
    char* aecres_filelist = const_cast<char*>(aecres_filelist_.c_str());

    // init input && output array
    const float* input_flat = input_tensor.flat<float>().data();
    short* input_data = new short[L];
    for (int i = 0; i < L; i++)
      input_data[i] = static_cast<short>(input_flat[i]);
    int outputdata_length[2];
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({1, L}),
                                                     &output_tensor));
    float* output_flat = output_tensor->flat<float>().data();
    short* output_data = new short[L];

    audio add_noise(sample_rate1);
    int ret;
    ret = add_noise.audio_pre_proc(
        input_data, L, output_data, &outputdata_length[0], if_add_rir_,
        rir_filelist, if_add_noise_, noise_filelist, snr_min_, snr_max_,
        if_add_aecres_, aecres_filelist);
    for (int i = 0; i < L; i++)
      output_flat[i] = static_cast<float>(output_data[i]);
    delete[] input_data;
    delete[] output_data;
  }

 private:
  float snr_min_;
  float snr_max_;
  bool if_add_rir_;
  bool if_add_noise_;
  bool if_add_aecres_;
  string rir_filelist_;
  string noise_filelist_;
  string aecres_filelist_;
};

REGISTER_KERNEL_BUILDER(Name("AddRirNoiseAecres").Device(DEVICE_CPU),
                        AddRirNoiseAecresOp);

}  // namespace delta
