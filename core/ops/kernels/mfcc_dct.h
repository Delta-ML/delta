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

// Basic minimal DCT class for MFCC speech processing.

#ifndef TENSORFLOW_CORE_KERNELS_MFCC_DCT_H_  // NOLINT
#define TENSORFLOW_CORE_KERNELS_MFCC_DCT_H_  // NOLINT

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "kernels/support_functions.h"

using namespace tensorflow;  // NOLINT
#define PI (3.141592653589793)

namespace delta {

class MfccDct {
 public:
  MfccDct();
  bool Initialize(int input_length, int coefficient_count);
  void Compute(const std::vector<double>& input,
               std::vector<double>* output) const;
  void set_coefficient_count(int coefficient_count);
  void set_cepstral_lifter(float cepstral_lifter);

 private:
  bool initialized_;
  int coefficient_count_;
  float cepstral_lifter_;
  int input_length_;
  std::vector<std::vector<double> > cosines_;
  std::vector<double> lifter_coeffs_;
  TF_DISALLOW_COPY_AND_ASSIGN(MfccDct);
};

}  // namespace delta

#endif  // DELTA_LAYERS_OPS_KERNELS_MFCC_DCT_H_
