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

#include "kernels/mel_spectrum.h"

#include <math.h>

#include "tensorflow/core/platform/logging.h"

namespace delta {

MelSpectrum::MelSpectrum() : initialized_(false) {}

MelSpectrum::~MelSpectrum() {
    for (int i=0; i<num_channels_;i++)
        std::vector<double>().swap(filterbank[i]);
}

bool MelSpectrum::Initialize(int input_length, double input_sample_rate,
                                   int output_channel_count,
                                   double lower_frequency_limit,
                                   double upper_frequency_limit) {
  num_channels_ = output_channel_count;
  sample_rate_ = input_sample_rate;
  input_length_ = input_length;
  int n_fft = (input_length_ - 1) * 2;

  if (num_channels_ < 1) {
    LOG(ERROR) << "Number of filterbank channels must be positive.";
    return false;
  }

  if (sample_rate_ <= 0) {
    LOG(ERROR) << "Sample rate must be positive.";
    return false;
  }

  if (input_length < 2) {
    LOG(ERROR) << "Input length must greater than 1.";
    return false;
  }

  if (lower_frequency_limit < 0) {
    LOG(ERROR) << "Lower frequency limit must be nonnegative.";
    return false;
  }

  if (upper_frequency_limit <= lower_frequency_limit) {
    LOG(ERROR) << "Upper frequency limit must be greater than "
               << "lower frequency limit.";
    return false;
  }

  const double mel_low = FreqToMel(lower_frequency_limit);
  const double mel_hi = FreqToMel(upper_frequency_limit);

  const double hz_per_sbin =
      0.5 * sample_rate_ / static_cast<double>(input_length_ - 1);
  start_index_ = static_cast<int>(1 + (lower_frequency_limit / hz_per_sbin));
  end_index_ = static_cast<int>(upper_frequency_limit / hz_per_sbin);

  // Init filterbank
  filterbank.resize(num_channels_);

  for (int i=0; i < num_channels_; i++){
    filterbank[i].resize(input_length_);
    for (int j=0; j < input_length_; j++)
        filterbank[i][j] = 0.0;
  }

  // Center frequency of each FFT bin
  std::vector<double> fftfreqs;
  fftfreqs.resize(input_length_);
  double fftbin = double(sample_rate_) / n_fft;
  for (int f = 0; f < input_length_; f++){
    fftfreqs[f] = double(fftbin * f);
  }

  // Center frequency of mel bands
  std::vector<double> mel_f, fdiff;
  mel_f.resize(num_channels_ + 2);
  fdiff.resize(num_channels_ + 1);
  double delta_mel = (mel_hi - mel_low) / (num_channels_ + 1);
  for (int i=0; i < num_channels_ + 2; i++){
    double tmp_mel = mel_low + i * delta_mel;
    double tmp_freq = MelToFreq(tmp_mel);
    mel_f[i] = tmp_freq;
  }

  for (int j = 0; j < num_channels_ + 1; j++){
    double tmp_diff = mel_f[j+1] - mel_f[j];
    fdiff[j] = tmp_diff;
  }

  std::vector<std::vector<double>> ramps;
  ramps.resize(num_channels_ + 2);

  for (int i = 0; i < num_channels_ + 2; i++){
    ramps[i].resize(input_length_);
    for (int j = 0; j < input_length_; j++){
        ramps[i][j] = mel_f[i] - fftfreqs[j];
    }
  }

  // Lower and upper slopes for all bins
  for (int i = 0; i < num_channels_; i++){
        std::vector<double> lower, upper;
        lower.resize(input_length_);
        upper.resize(input_length_);
        for (int j=0; j < input_length_; j++){
            lower[j] = -1 * ramps[i][j] / fdiff[i];
            upper[j] = ramps[i+2][j] / fdiff[i+1];
            if (lower[j] <= upper[j] && lower[j] > 0)
                filterbank[i][j] = lower[j];
            else if (lower[j] > upper[j] && upper[j] > 0)
                filterbank[i][j] = upper[j];
            else
                filterbank[i][j] = 0.0;
        }
    }

   // Slaney-style mel is scaled to be approx constant energy per channel
   std::vector<double> enorm;
   enorm.resize(num_channels_);
   for (int i=0; i < num_channels_; i++){
       enorm[i] = 2.0 / (mel_f[i+2] - mel_f[i]);
       for (int j=0; j < input_length_; j++){
            filterbank[i][j] = filterbank[i][j] * enorm[i];
       }
   }

  initialized_ = true;
  return true;
}

void MelSpectrum::Compute(const std::vector<double> &input,
                                std::vector<double> *output) const {
  if (!initialized_) {
    LOG(ERROR) << "Mel Filterbank not initialized.";
    return;
  }

  if (input.size() <= end_index_) {
    LOG(ERROR) << "Input too short to compute filterbank";
    return;
  }

  // Ensure output is right length and reset all values.
  output->assign(num_channels_, 0.0);

  for (int i=0; i < num_channels_; i++){
    double coeff = 0.0;
    for (int j = 0; j < input.size(); j++)
        coeff += input[j] * filterbank[i][j];
    if (coeff < 1e-10)
        coeff = 1e-10;
    (*output)[i] = log10(coeff);
  }
}

double MelSpectrum::FreqToMel(double freq) const {
    double f_min = 0.0;
    double f_sp = 200.0 / 3.0;
    double mel = (freq - f_min) / f_sp;
    double min_log_hz = 1000.0;
    double min_log_mel = (min_log_hz - f_min) / f_sp;
    double logstep = log(6.4) / 27.0;
    if (freq >= min_log_hz)
        mel = min_log_mel + log(freq / min_log_hz) / logstep;
    return mel;
}

double MelSpectrum::MelToFreq(double mel) const {
    double f_min = 0.0;
    double f_sp = 200.0 / 3.0;
    double freq = f_min + f_sp * mel;
    double min_log_hz = 1000.0;
    double min_log_mel = (min_log_hz - f_min) / f_sp;
    double logstep = log(6.4) / 27.0;
    if (mel >= min_log_mel)
        freq = min_log_hz * exp(logstep * (mel - min_log_mel));
    return freq;
}

}  // namespace delta
