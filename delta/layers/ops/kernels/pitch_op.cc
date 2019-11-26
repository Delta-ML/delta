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

#include "kernels/pitch.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace delta {

class PitchOp : public OpKernel {
 public:
  explicit PitchOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("snip_edges", &snip_edges_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("window_length", &window_length_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("frame_length", &frame_length_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("preemph_coeff", &preemph_coeff_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("min_f0", &min_f0_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_f0", &max_f0_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("soft_min_f0", &soft_min_f0_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("penalty_factor", &penalty_factor_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("lowpass_cutoff", &lowpass_cutoff_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("resample_freq", &resample_freq_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("delta_pitch", &delta_pitch_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("nccf_ballast", &nccf_ballast_));
     OP_REQUIRES_OK(context,
                   context->GetAttr("lowpass_filter_width", &lowpass_filter_width_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("upsample_filter_width", &upsample_filter_width_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_frames_latency", &max_frames_latency_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("frames_per_chunk", &frames_per_chunk_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("simulate_first_pass_online", &simulate_first_pass_online_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("recompute_frame", &recompute_frame_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("nccf_ballast_online", &nccf_ballast_online_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    OP_REQUIRES(context, input_tensor.dims() == 1,
                errors::InvalidArgument("input signal must be 1-dimensional",
                                        input_tensor.shape().DebugString()));
    const Tensor& sample_rate_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(sample_rate_tensor.shape()),
                errors::InvalidArgument(
                    "Input sample_rate should be a scalar tensor, got ",
                    sample_rate_tensor.shape().DebugString(), " instead."));
    const int32 sample_rate = sample_rate_tensor.scalar<int32>()();
    const float* input_flat = input_tensor.flat<float>().data();

    Tensor* output_tensor = nullptr;
    const int L = input_tensor.dim_size(0);
    int i_WinLen = static_cast<int>(window_length_ * sample_rate);
    int i_FrmLen = static_cast<int>(frame_length_ * sample_rate);
    int i_NumFrm = (L - i_WinLen) / i_FrmLen + 1;
    if (snip_edges_ == false)
        i_NumFrm = (L + i_FrmLen / 2) / i_FrmLen;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({i_NumFrm, 2}),
                                          &output_tensor));
    float* output_flat = output_tensor->flat<float>().data();
    PitchExtractionOptions pitch_opts;
    pitch_opts.set_samp_freq(static_cast<BaseFloat>(sample_rate));
    pitch_opts.set_frame_shift_ms(static_cast<BaseFloat>(frame_length_));
    pitch_opts.set_frame_length_ms(static_cast<BaseFloat>(window_length_));
    pitch_opts.set_preemph_coeff(static_cast<BaseFloat>(preemph_coeff_));
    pitch_opts.set_min_f0(static_cast<BaseFloat>(min_f0_));
    pitch_opts.set_max_f0(static_cast<BaseFloat>(max_f0_));
    pitch_opts.set_soft_min_f0(static_cast<BaseFloat>(soft_min_f0_));
    pitch_opts.set_penalty_factor(static_cast<BaseFloat>(penalty_factor_));
    pitch_opts.set_lowpass_cutoff(static_cast<BaseFloat>(lowpass_cutoff_));
    pitch_opts.set_resample_freq(static_cast<BaseFloat>(resample_freq_));
    pitch_opts.set_delta_pitch(static_cast<BaseFloat>(delta_pitch_));
    pitch_opts.set_nccf_ballast(static_cast<BaseFloat>(nccf_ballast_));
    pitch_opts.set_lowpass_filter_width(lowpass_filter_width_);
    pitch_opts.set_upsample_filter_width(upsample_filter_width_);
    pitch_opts.set_max_frames_latency(max_frames_latency_);
    pitch_opts.set_frames_per_chunk(frames_per_chunk_);
    pitch_opts.set_simulate_first_pass_online(simulate_first_pass_online_);
    pitch_opts.set_recompute_frame(recompute_frame_);
    pitch_opts.set_nccf_ballast_online(nccf_ballast_online_);
    pitch_opts.set_snip_edges(snip_edges_);
    vector<vector<BaseFloat>> features(i_NumFrm, vector<BaseFloat>(2));
    vector<BaseFloat> waveform(L);
    for (int i = 0; i < L; i++){
        waveform[i] = static_cast<BaseFloat>(input_flat[i]);
    }
    ComputeKaldiPitch(pitch_opts, waveform, &features);
    for(int j = 0; j < i_NumFrm; j++){
        for(int k = 0; k < 2; k++){
            output_flat[j * 2 + k] = static_cast<float>(features[j][k]);
        }
    }
   }

 private:
  float window_length_;
  float frame_length_;
  float preemph_coeff_;
  float min_f0_;
  float max_f0_;
  float soft_min_f0_;
  float penalty_factor_;
  float lowpass_cutoff_;
  float resample_freq_;
  float delta_pitch_;
  float nccf_ballast_;
  int lowpass_filter_width_;
  int upsample_filter_width_;
  int max_frames_latency_;
  int frames_per_chunk_;
  bool simulate_first_pass_online_;
  int recompute_frame_;
  bool nccf_ballast_online_;
  bool snip_edges_;
};

REGISTER_KERNEL_BUILDER(Name("Pitch").Device(DEVICE_CPU), PitchOp);

}  // namespace delta
