/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Basic class for applying a melFilter to a power/magnitude spectrum.

#ifndef DELTA_LAYERS_OPS_KERNELS_MEL_SPECTRUM_H_
#define DELTA_LAYERS_OPS_KERNELS_MEL_SPECTRUM_H_

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;  // NOLINT

namespace delta {

class MelSpectrum {
 public:
  MelSpectrum();
  ~MelSpectrum();
  bool Initialize(int input_length,  // Number of unique FFT bins fftsize/2+1.
                  double input_sample_rate, int output_channel_count,
                  double lower_frequency_limit, double upper_frequency_limit);

  void Compute(const std::vector<double>& input,
               std::vector<double>* output) const;

 private:
  double FreqToMel(double freq) const;
  double MelToFreq(double mel) const;
  bool initialized_;
  int num_channels_;
  double sample_rate_;
  int input_length_;
  std::vector<std::vector<double>> filterbank;
  int start_index_;  // Lowest FFT bin used to calculate mel spectrum.
  int end_index_;    // Highest FFT bin used to calculate mel spectrum.

  TF_DISALLOW_COPY_AND_ASSIGN(MelSpectrum);
};

}  // namespace delta
#endif  // DELTA_LAYERS_OPS_KERNELS_MEL_SPECTRUM_H_
