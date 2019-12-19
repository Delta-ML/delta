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

#include "kernels/fbank.h"

#include <math.h>

#include "tensorflow/core/platform/logging.h"

namespace delta {

const double kDefaultUpperFrequencyLimit = 4000;
const double kDefaultLowerFrequencyLimit = 20;
const double kFilterbankFloor = 1e-12;
const int kDefaultFilterbankChannelCount = 40;

Fbank::Fbank()
    : initialized_(false),
      lower_frequency_limit_(kDefaultLowerFrequencyLimit),
      upper_frequency_limit_(kDefaultUpperFrequencyLimit),
      filterbank_channel_count_(kDefaultFilterbankChannelCount) {}

Fbank::~Fbank() {}

bool Fbank::Initialize(int input_length, double input_sample_rate) {
  if (input_length < 1) {
    LOG(ERROR) << "Input length must be positive.";
    return false;
  }
  input_length_ = input_length;

  bool initialized = mel_filterbank_.Initialize(
      input_length, input_sample_rate, filterbank_channel_count_,
      lower_frequency_limit_, upper_frequency_limit_);
  initialized_ = initialized;
  return initialized;
}

void Fbank::Compute(const std::vector<double>& spectrogram_frame,
                    std::vector<double>* output) const {
  if (!initialized_) {
    LOG(ERROR) << "Fbank not initialized.";
    return;
  }

  output->resize(filterbank_channel_count_);
  mel_filterbank_.Compute(spectrogram_frame, output);
  for (int i = 0; i < output->size(); ++i) {
    double val = (*output)[i];
    if (val < kFilterbankFloor) {
      val = kFilterbankFloor;
    }
    (*output)[i] = log(val);
  }
}

}  // namespace delta
