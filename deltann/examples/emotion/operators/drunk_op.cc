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

#include "base.h"
#include "operator.h"

namespace inference {

namespace operators {

class DrunkOp : public Operator {
 public:
  DrunkOp(const OperatorConfig& config) : Operator(config) {
    BLOG << "Drunk Construct, _threshold is " << _threshold;
    std::vector<std::size_t> shape;
    _in = make_tuple("input", make_unique<Buffer<float>>(), shape);
    _out = make_tuple("frozen_softmax", make_unique<Buffer<float>>(), shape);
  }

  ~DrunkOp() {}

 private:
  int pre_process(Input& in) override {
    BLOG << "DrunkOp pre_process";
    BCHECK(in.size.size()) << "Input size is zero";
    BCHECK_EQ(in.inputs.size(), in.size.size())
        << "Error, Inputs size is " << in.inputs.size() << ", size is "
        << in.size.size();
    DBLOG << "_pcm_clip_duration_samples is " << _pcm_clip_duration_samples;

    // short --> float
    int batch = in.size.size();
    std::unique_ptr<Buffer<float>>& buffer = get_buffer(_in);
    buffer->resize(batch * _pcm_clip_duration_samples);
    buffer->zero();
    _pcm_size.clear();
    DBLOG << "buffer size is " << buffer->size();

    short* src = nullptr;
    float* dst = nullptr;
    for (int i = 0; i < batch; ++i) {
      src = in.inputs[i];
      dst = buffer->ptr() + i * _pcm_clip_duration_samples;
      BCHECK(src);
      BCHECK(in.size[i]);
      BCHECK_LE(in.size[i], _pcm_clip_duration_samples)
          << "The " << i << " in size is " << in.size[i]
          << ", _pcm_clip_duration_samples size is "
          << _pcm_clip_duration_samples;

#pragma simd
      for (int j = 0; j < in.size[i]; ++j) {
        dst[j] = src[j] * _scale;
      }
      _pcm_size.push_back(in.size[i]);
    }

    std::vector<std::size_t>& shape = get_shape(_in);  // must be reference
    shape.clear();
    shape.push_back(batch);
    shape.push_back(_pcm_clip_duration_samples);

    // set input and output
    BCHECK_EQ(0, set_input(_in)) << "set input error";
    BCHECK_EQ(0, set_output(_out)) << "set output error";

    return 0;
  }

  int post_process(Output& out) override {
    BLOG << "DrunkOp post_process";

    BCHECK_EQ(0, get_result(_out)) << "get result is error";
    std::string out_name = get_name(_out);
    std::vector<std::size_t>& shape = get_shape(_out);  // must be reference
    BCHECK(shape.size());
    int batch_size = shape[0];
    int row = shape[1];
    int dim = shape[2];
    DBLOG << "out shape is [" << batch_size << ", " << row << ", " << dim
          << "]";

    // get score
    std::vector<float> score_result;
    float* ptr = get_buffer(_out)->ptr();
    ++ptr;
    for (int n = 0; n < batch_size; ++n) {
      for (int i = 0; i < row; ++i) {
        score_result.push_back(*ptr);
        DBLOG << " i = " << i << ", " << *ptr;
        ptr += dim;
      }
    }

    // process
    int true_num = 0;
    int all_num = 0;

    for (int i = 0; i < batch_size; ++i) {
      int valid_len = row * _pcm_size[i] / _pcm_clip_duration_samples;
      DBLOG << "valid len is " << valid_len;
      for (int j = 0; j < valid_len; ++j) {
        if (score_result[i * row + j] > _threshold) {
          ++true_num;
        }
        ++all_num;
      }
    }
    DBLOG << "true num is " << true_num << ", all_num is " << all_num;

    bool& confidence = out.confidence;
    confidence = (true_num * 2) > all_num ? true : false;

    std::stringstream ss;
    ss << "{ \"model_type\":\"drunk\", ";  // emotion_type
    ss << "\"confidence\":" << confidence << ", ";
    ss << "\"score\":[";
#if 0
        if (0 != score_result.size()) {
            int score_size = score_result.size();
            for (int i = 0; i < score_size - 1; ++i) {
                ss << score_result[i] << ", ";
            }
            if (score_size - 1 >= 0) {
                ss << score_result[score_size - 1];
            }
        }
#endif
    ss << "] }";

    out.results.clear();
    (out.results).push_back(ss.str());

    return 0;
  }

  std::vector<std::size_t> _pcm_size;

  DATA_TYPE<float> _in;
  DATA_TYPE<float> _out;
};

RESISTER_OPERATOR(drunk, DrunkOp);

}  // namespace operators

}  // namespace inference
