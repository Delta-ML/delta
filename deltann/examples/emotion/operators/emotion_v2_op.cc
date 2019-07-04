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
#include "dynload/feature_extraction_wrapper.h"
#include "operator.h"

namespace inference {

namespace operators {

using namespace FeatureExtraction;

namespace {}  // end namespace

class EmotionV2Op : public Operator {
 public:
  EmotionV2Op(const OperatorConfig& config) : Operator(config) {
    BLOG << "EmotionV2 Construct, _threshold is " << _threshold;
    BLOG << "EmotionV2 window " << _window << "ms";
    BLOG << "EmotionV2 step " << _step << "ms";

    std::vector<std::size_t> shape;
    _in = make_tuple("inputs", make_unique<Buffer<float>>(), shape);
    _out = make_tuple("softmax_output", make_unique<Buffer<float>>(), shape);

    float window = _window / _scalar;  // 0.025s
    float step = _step / _scalar;      // 0.01s
    bool delta = true;
    _fbank =
        FBANK_CREATE(window, step, _sample_rate, _filter_num, delta);  // init
  }

  ~EmotionV2Op() {
    if (!_fbank) {
      FBANK_DESTROY(_fbank);
      _fbank = nullptr;
    }
  }

 private:
  int pre_process(Input& in) override {
    DBLOG << "EmotionV2Op pre_process, this is " << this;

    int pcm_sample_count = in.size[0];
    DBLOG << "In emotion_impl pcm_sample_count is " << pcm_sample_count;

    // feature extraction
    std::vector<std::vector<double>> feature_out;
    BCHECK(in.input) << "in.input is nullptr";
    BCHECK(_fbank) << "_fbank is nullptr";
    FBANK_EXTRACTFEAT(_fbank, in.input, pcm_sample_count, &feature_out,
                      true);  // ExtractFeat
    BCHECK(feature_out.size());
    BCHECK(feature_out[0].size());

    int batch_size = feature_out.size();
    int dim = feature_out[0].size();
    BCHECK_EQ(dim, _filter_num * _channel);
    DBLOG << "In emotion_impl batch_size is " << batch_size << ", dim is "
          << dim;

    std::unique_ptr<Buffer<float>>& buffer = get_buffer(_in);

    int batch_num = (batch_size + (_batch - 1)) / _batch;
    DBLOG << "batch_num is " << batch_num;
    buffer->resize(batch_num * _batch * dim);
    buffer->zero();

    DBLOG << "buffer size is " << buffer->size();

    float* pcm_dst = buffer->ptr();
    for (int i = 0; i < batch_size; ++i) {
      std::vector<double>& v = feature_out[i];
      for (int j = 0; j < dim; ++j) {
        pcm_dst[i * dim + j] = v[j];  // double --> float
      }
    }

    std::vector<std::size_t>& shape = get_shape(_in);  // must be reference
    shape.clear();
    shape.push_back(batch_num);
    shape.push_back(_batch);
    shape.push_back(_filter_num);
    shape.push_back(_channel);

    // set input and output
    BCHECK_EQ(0, set_input(_in)) << "set input error";
    BCHECK_EQ(0, set_output(_out)) << "set output error";

    return 0;
  }

  int post_process(Output& out) override {
    DBLOG << "EmotionV2Op post_process";

    BCHECK_EQ(0, get_result(_out));
    std::vector<std::size_t>& shape = get_shape(_out);  // must be reference
    BCHECK(shape.size()) << "out shape is " << shape.size();
    int batch_size = shape[0];
    int dim = shape[1];
    DBLOG << "out shape is [" << batch_size << ", " << dim << "]";

    std::vector<float> score_result;
    float* ptr = get_buffer(_out)->ptr();
    float sum = 0.0f;
    for (int n = 0; n < batch_size; ++n) {
      DBLOG << "[" << *ptr << ", " << *(ptr + 1) << "]";
      score_result.push_back(*(ptr + 1));
      sum += *(ptr + 1);
      ptr += dim;
    }
    sum /= batch_size;
    DBLOG << "score_result size is " << score_result.size();

    out.confidence = sum > _threshold ? true : false;

    std::stringstream ss;

    ss << "{ \"model_type\":\"emotion_v2\", ";  // emotion_type
    ss << "\"confidence\":" << out.confidence << ", ";
    ss << "\"score\": [";
    for (int n = 0; n < batch_size - 1; ++n) {
      ss << score_result[n] << ", ";
    }
    ss << score_result[batch_size - 1];
    ss << "] }";
    out.results.clear();
    (out.results).push_back(ss.str());

    return 0;
  }

  FilterBank* _fbank = nullptr;
  int _filter_num = 40;
  int _channel = 3;
  int _batch = 3000;

  DATA_TYPE<float> _in;
  DATA_TYPE<float> _out;
};

RESISTER_OPERATOR(emotion_v2, EmotionV2Op);

}  // namespace operators

}  // namespace inference
