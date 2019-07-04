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

namespace {

int smooth_score(std::vector<float>& result, std::vector<float>& avg_result) {
  const int history_num = 8;  //(1s)
  float sum = 0.0f;
  int size = result.size();

  // smooth
  // std::vector<float> avg_result;
  int i = 0;
  for (; i < history_num && i < size; ++i) {
    sum += result[i];
    avg_result.push_back(sum / (i + 1));
  }

  for (; i < size; ++i) {
    sum += result[i];
    sum -= result[i - history_num];
    avg_result.push_back(sum / history_num);
  }

  return 0;
}

int process_score(std::vector<float>& avg_result, const float threshold,
                  std::vector<float>& score_result, bool need_origin_score,
                  Output& out) {
  // postprocess score
  int frame_shift_in_ms = 40;                           // ms
  int num_frame_per_second = 1000 / frame_shift_in_ms;  // 25
  int confict_time = 10;                                // s
  int confict_window = confict_time * num_frame_per_second;

  std::vector<float> confict_point;
  std::vector<float> confict_dur;
  std::vector<float> avg_confidences;

  int size = avg_result.size();
  int s = 0;
  int e = 0;
  int dur = 0;
  float sum_confidence = 0.0f;
  for (; s < size;) {
    if (avg_result[s] < threshold) {
      ++s;
    } else {
      e = s;
      for (; e < size; ++e) {
        sum_confidence += avg_result[e];
        if (avg_result[e] < threshold) {
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
  std::stringstream ss;
  ss << "{ \"model_type\":\"emotion\", ";  // emotion_type
  ss << "\"confidence\":" << confidence << ", ";
  ss << "\"time\":[";
  if (0 != c_size) {
    for (int c = 0; c < c_size - 1; ++c) {
      ss << " [" << confict_point[c] << ", " << confict_dur[c] << ", "
         << avg_confidences[c] << "], ";
    }
    if (c_size - 1 >= 0) {
      ss << " [" << confict_point[c_size - 1] << ", " << confict_dur[c_size - 1]
         << ", " << avg_confidences[c_size - 1] << "] ";
    }
  }

  if (need_origin_score) {
    int score_size = score_result.size();
    ss << "], \"score\":[";
    for (int i = 0; i < score_size - 1; ++i) {
      ss << score_result[i] << ", ";
    }
    ss << score_result[score_size - 1] << "] }";
  } else {
    ss << "] }";
  }
  confict_time_offset = ss.str();
  out.results.clear();
  (out.results).push_back(confict_time_offset);

  DBLOG << "confidence is " << confidence;
  DBLOG << "time offset  " << out.results[0];

  return 0;
}

}  // end namespace

class EmotionOp : public Operator {
 public:
  EmotionOp(const OperatorConfig& config) : Operator(config) {
    BLOG << "Emotion Construct, _threshold is " << _threshold;
    std::vector<std::size_t> shape;
    _in = make_tuple("input", make_unique<Buffer<float>>(), shape);
    _out = make_tuple("frozen_softmax", make_unique<Buffer<float>>(), shape);
  }

  ~EmotionOp() {}

 private:
  int pre_process(Input& in) override {
    DBLOG << "EmotionOp pre_process, this is " << this;

    std::size_t pcm_sample_count = in.size[0];
    DBLOG << "In emotion_impl pcm_sample_count is " << pcm_sample_count;

    int batch_size = (pcm_sample_count + _pcm_clip_duration_samples - 1) /
                     _pcm_clip_duration_samples;
    DBLOG << "In emotion_impl batch_size is " << batch_size;
    if (batch_size < 0) {
      BFATAL << "Error : batch_size is 0 ";
      return -1;
    }
    std::unique_ptr<Buffer<float>>& buffer = get_buffer(_in);
    buffer->resize(batch_size * _pcm_clip_duration_samples);
    DBLOG << "buffer size is " << buffer->size();

    // pcm short --> pcm float
    pcm_short_to_float(in, buffer.get());
    DBLOG << "In emotion_impl _pcm_clip_duration_samples "
          << _pcm_clip_duration_samples;

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
    DBLOG << "EmotionOp post_process";

    BCHECK_EQ(0, get_result(_out));
    std::vector<std::size_t>& shape = get_shape(_out);  // must be reference
    BCHECK(shape.size());
    int batch_size = shape[0];
    int row = shape[1];
    int dim = shape[2];
    DBLOG << "out shape is [" << batch_size << ", " << row << ", " << dim
          << "]";

    std::vector<float> score_result;
    float* ptr = get_buffer(_out)->ptr();
    ++ptr;
    for (int n = 0; n < batch_size; ++n) {
      for (int i = 0; i < row; ++i) {
        score_result.push_back(*ptr);
        ptr += dim;
      }
    }
    DBLOG << "score_result size is " << score_result.size();

    std::vector<float> avg_result;
    smooth_score(score_result, avg_result);

    process_score(avg_result, _threshold, score_result, _need_origin_score,
                  out);

    return 0;
  }

  DATA_TYPE<float> _in;
  DATA_TYPE<float> _out;
};

RESISTER_OPERATOR(emotion, EmotionOp);

}  // namespace operators

}  // namespace inference
