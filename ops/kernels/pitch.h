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

#ifndef DELTA_LAYERS_OPS_KERNELS_PITCH_H_
#define DELTA_LAYERS_OPS_KERNELS_PITCH_H_

#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "kernels/support_functions.h"
#include "kernels/resample.h"

using namespace tensorflow;  // NOLINT

namespace delta {

struct PitchExtractionOptions {
  // FrameExtractionOptions frame_opts;
  BaseFloat samp_freq;          // sample frequency in hertz
  BaseFloat frame_shift_ms;     // in milliseconds.
  BaseFloat frame_length_ms;    // in milliseconds.
  BaseFloat preemph_coeff;      // Preemphasis coefficient. [use is deprecated.]
  BaseFloat min_f0;             // min f0 to search (Hz)
  BaseFloat max_f0;             // max f0 to search (Hz)
  BaseFloat soft_min_f0;        // Minimum f0, applied in soft way, must not
                                // exceed min-f0
  BaseFloat penalty_factor;     // cost factor for FO change
  BaseFloat lowpass_cutoff;     // cutoff frequency for Low pass filter
  BaseFloat resample_freq;      // Integer that determines filter width when
                                // upsampling NCCF
  BaseFloat delta_pitch;        // the pitch tolerance in pruning lags
  BaseFloat nccf_ballast;       // Increasing this factor reduces NCCF for
                                // quiet frames, helping ensure pitch
                                // continuity in unvoiced region
  int lowpass_filter_width;   // Integer that determines filter width of
                                // lowpass filter
  int upsample_filter_width;  // Integer that determines filter width when
                                // upsampling NCCF

  int max_frames_latency;
  int frames_per_chunk;

  bool simulate_first_pass_online;

  int recompute_frame;

  bool nccf_ballast_online;
  bool snip_edges;
  PitchExtractionOptions():
      samp_freq(16000),
      frame_shift_ms(10.0),
      frame_length_ms(25.0),
      preemph_coeff(0.0),
      min_f0(50),
      max_f0(400),
      soft_min_f0(10.0),
      penalty_factor(0.1),
      lowpass_cutoff(1000),
      resample_freq(4000),
      delta_pitch(0.005),
      nccf_ballast(7000),
      lowpass_filter_width(1),
      upsample_filter_width(5),
      max_frames_latency(0),
      frames_per_chunk(0),
      simulate_first_pass_online(false),
      recompute_frame(500),
      nccf_ballast_online(false),
      snip_edges(true) { }

  void set_samp_freq(BaseFloat sample_rate) { samp_freq = sample_rate; }
  void set_frame_shift_ms(BaseFloat frame_shift) { frame_shift_ms = frame_shift * 1000; }
  void set_frame_length_ms(BaseFloat window_length) { frame_length_ms = window_length * 1000; }
  void set_preemph_coeff(BaseFloat preemph_coeff_) { preemph_coeff = preemph_coeff_; }
  void set_min_f0(BaseFloat min_f0_) { min_f0 = min_f0_; }
  void set_max_f0(BaseFloat max_f0_) { max_f0 = max_f0_; }
  void set_soft_min_f0(BaseFloat soft_min_f0_) { soft_min_f0 = soft_min_f0_; }
  void set_penalty_factor(BaseFloat penalty_factor_) { penalty_factor = penalty_factor_; }
  void set_lowpass_cutoff(BaseFloat lowpass_cutoff_) { lowpass_cutoff = lowpass_cutoff_; }
  void set_resample_freq(BaseFloat resample_freq_) { resample_freq = resample_freq_; }
  void set_delta_pitch(BaseFloat delta_pitch_) { delta_pitch = delta_pitch_; }
  void set_nccf_ballast(BaseFloat nccf_ballast_) { nccf_ballast = nccf_ballast_; }
  void set_lowpass_filter_width(int lowpass_filter_width_) { lowpass_filter_width = lowpass_filter_width_; }
  void set_upsample_filter_width(int upsample_filter_width_) { upsample_filter_width = upsample_filter_width_; }
  void set_max_frames_latency(int max_frames_latency_) { max_frames_latency = max_frames_latency_; }
  void set_frames_per_chunk(int frames_per_chunk_) { frames_per_chunk = frames_per_chunk_; }
  void set_simulate_first_pass_online(bool simulate_first_pass_online_) { simulate_first_pass_online = simulate_first_pass_online_; }
  void set_recompute_frame(int recompute_frame_) { recompute_frame = recompute_frame_; }
  void set_nccf_ballast_online(bool nccf_ballast_online_) { nccf_ballast_online = nccf_ballast_online_; }
  void set_snip_edges(bool snip_edges_) { snip_edges = snip_edges_; }

  int NccfWindowSize() const {
    return static_cast<int>(resample_freq * frame_length_ms / 1000.0);
  }
  /// Returns the window-shift in samples, after resampling.
  int NccfWindowShift() const {
    return static_cast<int>(resample_freq * frame_shift_ms / 1000.0);
  }
};

class OnlinePitchFeatureImpl;

class OnlineFeatureInterface {
 public:
  virtual int Dim() const = 0; /// returns the feature dimension.
  virtual int NumFramesReady() const = 0;

  virtual bool IsLastFrame(int frame) const = 0;
  virtual void GetFrame(int frame, vector<BaseFloat> *feat) = 0;

  virtual void GetFrames(const std::vector<int> &frames,
                         vector<vector<BaseFloat> > *feats) {
    for (int i = 0; i < frames.size(); i++) {
      vector<BaseFloat> feat = (*feats)[i];
      GetFrame(frames[i], &feat);
    }
  }

  virtual BaseFloat FrameShiftInSeconds() const = 0;
  virtual ~OnlineFeatureInterface() { }

};

class OnlineBaseFeature: public OnlineFeatureInterface {
 public:
  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const vector<BaseFloat> &waveform) = 0;

  virtual void InputFinished() = 0;
};



// Note: to start on a new waveform, just construct a new version
// of this object.
class OnlinePitchFeature: public OnlineBaseFeature {
 public:
  explicit OnlinePitchFeature(const PitchExtractionOptions &opts);

  virtual int Dim() const { return 2; /* (NCCF, pitch) */ }

  virtual int NumFramesReady() const;

  virtual BaseFloat FrameShiftInSeconds() const;

  virtual bool IsLastFrame(int frame) const;

  /// Outputs the two-dimensional feature consisting of (pitch, NCCF).  You
  /// should probably post-process this using class OnlineProcessPitch.
  virtual void GetFrame(int frame, vector<BaseFloat> *feat);

  virtual void AcceptWaveform(BaseFloat sampling_rate,
                              const vector<BaseFloat> &waveform);

  virtual void InputFinished();

  virtual ~OnlinePitchFeature();

 private:
  OnlinePitchFeatureImpl *impl_;
};

/// This function extracts (pitch, NCCF) per frame.
void ComputeKaldiPitch(const PitchExtractionOptions &opts,
                       const vector<BaseFloat> &wave,
                       vector<vector<BaseFloat>> *output);

}  // namespace delta
#endif  // DELTA_LAYERS_OPS_KERNELS_PITCH_H_
