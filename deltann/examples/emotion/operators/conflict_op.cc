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

/**
 * @file conflict_op.cc
 * @brief This conflict model train by liwubo, offer by jiangdongwei
 * @author gaoyonghu
 * @version 0.1
 * @date 2019-01-14
 * */

#include "base.h"
#include "operator.h"

namespace inference {

namespace operators {

namespace {

int process_score(std::vector<float>& score_result, const float threshold,
                  Output& out) {
  // postprocess score
  int frame_shift_in_ms = 10000;                     // ms
  int num_frame_per_second = 1 / frame_shift_in_ms;  // 25
  int confict_time = 2;                              // 20s
  int confict_window = confict_time * num_frame_per_second;

  std::vector<float> confict_point;
  std::vector<float> confict_dur;
  std::vector<float> avg_confidences;

  int size = score_result.size();
  int s = 0;
  int e = 0;
  int dur = 0;
  float sum_confidence = 0.0f;
  for (; s < size;) {
    if (score_result[s] < threshold) {
      ++s;
    } else {
      e = s;
      for (; e < size; ++e) {
        sum_confidence += score_result[e];
        if (score_result[e] < threshold) {
          break;
        }
      }

      if (e - s >= confict_window) {
        confict_point.push_back(s * frame_shift_in_ms);
        confict_dur.push_back((e - s) * frame_shift_in_ms);
        avg_confidences.push_back(sum_confidence / (e - s));
        sum_confidence = 0.0f;
        s = e;
      } else {
        sum_confidence = 0.0f;
        s = e;
      }
    }
  }

  bool& confidence = out.confidence;
  std::string confict_time_offset;

  int c_size = confict_point.size();
  confidence = confict_point.size() > 0 ? true : false;

  int score_size = score_result.size();
  std::stringstream ss;
  ss << "{ \"model_type\":\"conflict\", ";  // model_type
  ss << "\"confidence\":" << confidence << ", ";
  ss << "\"score\":[";
  if (0 != score_size) {
    for (int c = 0; c < score_size - 1; ++c) {
      ss << score_result[c] << ", ";
    }
    if (score_size - 1 >= 0) {
      ss << score_result[score_size - 1];
    }
  }
  ss << "] }";
  confict_time_offset = ss.str();
  out.results.clear();
  (out.results).push_back(confict_time_offset);

  DBLOG << "confidence is " << confidence;
  DBLOG << "score  " << out.results[0];

  return 0;
}

}  // end namespace

class ConflictOp : public Operator {
 public:
  ConflictOp(const OperatorConfig& config) : Operator(config) {
    BLOG << "Conflict Construct, _threshold is " << _threshold;
    std::vector<std::size_t> shape;
    _in = make_tuple("input", make_unique<Buffer<float>>(), shape);
    _out = make_tuple("frozen_softmax", make_unique<Buffer<float>>(), shape);
  }

  ~ConflictOp() {}

 private:
  int pre_process(Input& in) override {
    DBLOG << "ConflictOp pre_process, this is " << this;

    std::size_t pcm_sample_count = in.size[0];
    DBLOG << "In pcm_sample_count is " << pcm_sample_count;

    int batch_size = (pcm_sample_count + _pcm_clip_duration_samples - 1) /
                     _pcm_clip_duration_samples;
    DBLOG << "In batch_size is " << batch_size;
    if (batch_size < 0) {
      BFATAL << "Error : batch_size is 0 ";
      return -1;
    }
    std::unique_ptr<Buffer<float>>& buffer = get_buffer(_in);
    buffer->resize(batch_size * _pcm_clip_duration_samples);
    DBLOG << "buffer size is " << buffer->size();

    // pcm short --> pcm float
    pcm_short_to_float(in, buffer.get());
    DBLOG << "In _pcm_clip_duration_samples " << _pcm_clip_duration_samples;

    int leftover = pcm_sample_count % _pcm_clip_duration_samples;
    if (0 != leftover) {
      DBLOG << "leftover data is " << leftover << ", padding len is "
            << (_pcm_clip_duration_samples - leftover);
      memset(buffer->ptr() + pcm_sample_count, 0,
             (_pcm_clip_duration_samples - leftover) * sizeof(float));
    }

    std::vector<std::size_t>& shape = get_shape(_in);  // must be reference
    shape.clear();
    shape.push_back(batch_size);
    shape.push_back(_pcm_clip_duration_samples);

    // set input and output
    BCHECK_EQ(0, set_input(_in)) << "set input error";
    BCHECK_EQ(0, set_output(_out)) << "set output error";

    return 0;
  }

  int post_process(Output& out) override {
    DBLOG << "ConflictOp post_process";

    BCHECK_EQ(0, get_result(_out));
    std::vector<std::size_t>& shape = get_shape(_out);  // must be reference
    BCHECK(shape.size());
    int batch_size = shape[0];
    int dim = shape[1];
    DBLOG << "out shape is [" << batch_size << ", " << dim << "]";

    std::vector<float> score_result;
    float* ptr = get_buffer(_out)->ptr();
    ++ptr;
    for (int n = 0; n < batch_size; ++n) {
      score_result.push_back(*ptr);
      ptr += dim;
    }
    DBLOG << "score_result size is " << score_result.size();

    process_score(score_result, _threshold, out);

    return 0;
  }

  DATA_TYPE<float> _in;
  DATA_TYPE<float> _out;
};

RESISTER_OPERATOR(conflict, ConflictOp);

}  // namespace operators

}  // namespace inference
