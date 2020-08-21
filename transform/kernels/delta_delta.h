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

#ifndef DELTA_LAYERS_OPS_KERNELS_DELTA_DELTA_H_
#define DELTA_LAYERS_OPS_KERNELS_DELTA_DELTA_H_

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;  // NOLINT

namespace delta {

// This class provides a low-level function to compute delta features.
// The function takes as input a matrix of features and a frame index
// that it should compute the deltas on.  It puts its output in an object
// of type VectorBase, of size (original-feature-dimension) * (opts.order+1).
// This is not the most efficient way to do the computation, but it's
// state-free and thus easier to understand
class DeltaDelta {
 public:
  DeltaDelta();
  ~DeltaDelta();

  bool Initialize(int order, int window);

  // Input is a single feature frame. Output is populated with
  // the feature, delta, delta-delta values.
  void Compute(const Tensor& input_feats, int frame,
               std::vector<double>* output) const;
  void Compute_frame(const std::vector<std::vector<float>> &input_feats, int frame,
                std::vector<float>* output) const;

  void set_order(int order) {
    CHECK(!initialized_) << "Set order before calling Initialize.";
    order_ = order;
  }

  void set_window(int window) {
    CHECK(!initialized_) << "Set window before calling Initialize.";
    window_ = window;
  }

 private:
  int order_;
  // e.g. 2; controls window size (window size is 2*window + 1)
  // the behavior at the edges is to replicate the first or last frame.
  // this is not configurable.
  int window_;

  // a scaling window for each of the orders, including zero:
  // multiply the features for each dimension by this window.
  std::vector<std::vector<double> > scales_;

  bool initialized_;
  TF_DISALLOW_COPY_AND_ASSIGN(DeltaDelta);
};  // class DeltaDelta

}  // namespace delta

#endif  // DELTA_LAYERS_OPS_KERNELS_DELTA_DELTA_H_
