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

#ifndef DELTA_LAYERS_OPS_KERNELS_FBANK_H_
#define DELTA_LAYERS_OPS_KERNELS_FBANK_H_

#include <vector>  // NOLINT

#include "kernels/mfcc_mel_filterbank.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;  // NOLINT

namespace delta {

// logfbank
class Fbank {
 public:
  Fbank();
  ~Fbank();
  bool Initialize(int input_length, double input_sample_rate);
  // Input is a single squared-magnitude spectrogram frame. The input spectrum
  // is converted to linear magnitude and weighted into bands using a
  // triangular mel filterbank. Output is populated with the lowest
  // fbank_channel_count
  // of these values.
  void Compute(const std::vector<double>& spectrogram_frame,
               std::vector<double>* output) const;

  void set_upper_frequency_limit(double upper_frequency_limit) {
    CHECK(!initialized_) << "Set frequency limits before calling Initialize.";
    upper_frequency_limit_ = upper_frequency_limit;
  }

  void set_lower_frequency_limit(double lower_frequency_limit) {
    CHECK(!initialized_) << "Set frequency limits before calling Initialize.";
    lower_frequency_limit_ = lower_frequency_limit;
  }

  void set_filterbank_channel_count(int filterbank_channel_count) {
    CHECK(!initialized_) << "Set channel count before calling Initialize.";
    filterbank_channel_count_ = filterbank_channel_count;
  }

 private:
  MfccMelFilterbank mel_filterbank_;
  int input_length_;
  bool initialized_;
  double lower_frequency_limit_;
  double upper_frequency_limit_;
  int filterbank_channel_count_;
  TF_DISALLOW_COPY_AND_ASSIGN(Fbank);
};  // class Fbank

}  // namespace delta

#endif  // DELTA_LAYERS_OPS_KERNELS_FBANK_H_
