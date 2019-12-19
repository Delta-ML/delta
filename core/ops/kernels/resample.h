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


#ifndef DELTA_LAYERS_OPS_KERNELS_RESAMPLE_H_
#define DELTA_LAYERS_OPS_KERNELS_RESAMPLE_H_

#include <cassert>
#include <cstdlib>
#include <string>
#include <stdlib.h>
#include <vector>
#include <assert.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
using namespace tensorflow;  // NOLINT

using namespace std;
#include "kernels/support_functions.h"

namespace delta {

class ArbitraryResample {
 public:
  ArbitraryResample(int num_samples_in,
                    BaseFloat samp_rate_hz,
                    BaseFloat filter_cutoff_hz,
                    const vector<BaseFloat> &sample_points_secs,
                    int num_zeros);

  int NumSamplesIn() const { return num_samples_in_; }

  int NumSamplesOut() const { return weights_.size(); }
  void Resample(const std::vector<vector<BaseFloat> > &input,
                std::vector<vector<BaseFloat> > *output) const;

  void Resample(const vector<BaseFloat> &input,
                vector<BaseFloat> *output) const;
 private:
  void SetIndexes(const vector<BaseFloat> &sample_points);

  void SetWeights(const vector<BaseFloat> &sample_points);

  BaseFloat FilterFunc(BaseFloat t) const;

  int num_samples_in_;
  BaseFloat samp_rate_in_;
  BaseFloat filter_cutoff_;
  int num_zeros_;

  std::vector<int> first_index_;
  std::vector<vector<BaseFloat> > weights_;
};

class LinearResample {
 public:
  LinearResample(int samp_rate_in_hz,
                 int samp_rate_out_hz,
                 BaseFloat filter_cutoff_hz,
                 int num_zeros);

  void Resample(const vector<BaseFloat> &input,
                bool flush,
                vector<BaseFloat> *output);

  void Reset();

  //// Return the input and output sampling rates (for checks, for example)
  inline int GetInputSamplingRate() { return samp_rate_in_; }
  inline int GetOutputSamplingRate() { return samp_rate_out_; }
 private:
  int GetNumOutputSamples(int input_num_samp, bool flush) const;

  inline void GetIndexes(int samp_out,
                         int *first_samp_in,
                         int *samp_out_wrapped) const;

  void SetRemainder(const vector<BaseFloat> &input);

  void SetIndexesAndWeights();

  BaseFloat FilterFunc(BaseFloat) const;

  // The following variables are provided by the user.
  int samp_rate_in_;
  int samp_rate_out_;
  BaseFloat filter_cutoff_;
  int num_zeros_;

  int input_samples_in_unit_;
  int output_samples_in_unit_;

  std::vector<int> first_index_;
  std::vector<vector<BaseFloat> > weights_;

  int input_sample_offset_;
  int output_sample_offset_;
  vector<BaseFloat> input_remainder_;
};

void ResampleWaveform(BaseFloat orig_freq, const vector<BaseFloat> &wave,
                      BaseFloat new_freq, vector<BaseFloat> *new_wave);

inline void DownsampleWaveForm(BaseFloat orig_freq, const vector<BaseFloat> &wave,
                               BaseFloat new_freq, vector<BaseFloat> *new_wave) {
  ResampleWaveform(orig_freq, wave, new_freq, new_wave);
}

}  // namespace delta
#endif  // DELTA_LAYERS_OPS_KERNELS_RESAMPLE_H_
