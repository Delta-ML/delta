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

#include "../../../kernels/pitch.h"
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
    OP_REQUIRES_OK(context,
                   context->GetAttr("pitch_scale", &pitch_scale_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("pov_scale", &pov_scale_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("pov_offset", &pov_offset_));
    OP_REQUIRES_OK(context, context->GetAttr("delta_pitch_scale", &delta_pitch_scale_));
    OP_REQUIRES_OK(context, context->GetAttr("delta_pitch_noise_stddev", &delta_pitch_noise_stddev_));
    OP_REQUIRES_OK(context, context->GetAttr("normalization_left_context", &normalization_left_context_));
    OP_REQUIRES_OK(context, context->GetAttr("normalization_right_context", &normalization_right_context_));
    OP_REQUIRES_OK(context, context->GetAttr("delta_window", &delta_window_));
    OP_REQUIRES_OK(context, context->GetAttr("delay", &delay_));
    OP_REQUIRES_OK(context, context->GetAttr("add_pov_feature", &add_pov_feature_));
    OP_REQUIRES_OK(context, context->GetAttr("add_normalized_log_pitch", &add_normalized_log_pitch_));
    OP_REQUIRES_OK(context, context->GetAttr("add_delta_pitch", &add_delta_pitch_));
    OP_REQUIRES_OK(context, context->GetAttr("add_raw_log_pitch", &add_raw_log_pitch_));
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

//    Process Pitch features
    ProcessPitchOptions process_opts;
    process_opts.set_pitch_scale(static_cast<BaseFloat>(pitch_scale_));
    process_opts.set_pov_scale(pov_scale_);
    process_opts.set_pov_offset(pov_offset_);
    process_opts.set_delta_pitch_scale(delta_pitch_scale_);
    process_opts.set_delta_pitch_noise_stddev(delta_pitch_noise_stddev_);
    process_opts.set_normalization_left_context(normalization_left_context_);
    process_opts.set_normalization_right_context(normalization_right_context_);
    process_opts.set_delta_window(delta_window_);
    process_opts.set_delay(delay_);
    process_opts.set_add_pov_feature(add_pov_feature_);
    process_opts.set_add_normalized_log_pitch(add_normalized_log_pitch_);
    process_opts.set_add_delta_pitch(add_delta_pitch_);
    process_opts.set_add_raw_log_pitch(add_raw_log_pitch_);

    vector<vector<BaseFloat>> prcocessd_features(i_NumFrm, vector<BaseFloat>(3));
    ProcessPitch(process_opts, features, &prcocessd_features);
    int dim = prcocessd_features[0].size();
    OP_REQUIRES_OK(
    context, context->allocate_output(0, TensorShape({i_NumFrm, dim}),
                                      &output_tensor));
    float* output_flat = output_tensor->flat<float>().data();
     for(int n = 0; n < i_NumFrm; n++){
        for(int m = 0; m < dim; m++){
            output_flat[n * dim + m] = static_cast<float>(prcocessd_features[n][m]);
        }
     }

    for (int i = 0; i < i_NumFrm; i++)
    vector<BaseFloat>().swap(features[i]);
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
  float pitch_scale_;
  float pov_scale_;
  float pov_offset_;
  float delta_pitch_scale_;
  float delta_pitch_noise_stddev_;
  int normalization_left_context_;
  int normalization_right_context_;
  int delta_window_;
  int delay_;
  bool add_pov_feature_;
  bool add_normalized_log_pitch_;
  bool add_delta_pitch_;
  bool add_raw_log_pitch_;
};

REGISTER_KERNEL_BUILDER(Name("Pitch").Device(DEVICE_CPU), PitchOp);

}  // namespace delta
