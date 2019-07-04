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

#include "CFrontend.h"
#include "base.h"
#include "operator.h"

namespace inference {

namespace operators {

namespace {}  // end namespace

class AttentionOp : public Operator {
 public:
  AttentionOp(const OperatorConfig& config)
      : Operator(config),
        _pcm_buffer(nullptr),
        _feat_buffer(nullptr),
        _cmvn_mean_array(nullptr),
        _cmvn_std_array(nullptr),
        _feat_ext(nullptr),
        _feat_ext_sample_rate(0),
        _feat_ext_package_size(0),
        _feat_ext_frame_dim(0),
        _feat_frame_count(0) {
    BLOG << "Attention Construct";
    DBLOG << "_pcm_clip_duration_samples : " << _pcm_clip_duration_samples;
    DBLOG << "_pcm_clip_stride_samples : " << _pcm_clip_stride_samples;

    init_feat_ext();
    init_cmvn_array();
    init_char_list();

    // init buffer
    _pcm_buffer = make_unique<Buffer<short>>();
    //_feat_buffer = make_unique<Buffer<float>>();

    // set input and ouput name
    std::vector<std::size_t> shape;
    _in_seq =
        std::make_tuple("e2e_model/Model_1/input_seq",
                        make_unique<Buffer<float>>(), shape);  // input_seq_name
    _in_len =
        std::make_tuple("e2e_model/Model_1/input_len",
                        make_unique<Buffer<int>>(), shape);  // input_seq_name
    _out = std::make_tuple(
        "e2e_model/Model_1/attn_decoder/LASDecoder/decoding_output",
        make_unique<Buffer<int>>(), shape);
  }

  ~AttentionOp() {
    if (_feat_ext) {
      _feat_ext->Unit();
    }
  }

 private:
  int pre_process(Input& in) override {
    DBLOG << "AttentionOp pre_process, this is " << this;
    // TODO ...
    BCHECK(in.input);
    BCHECK_EQ(1, in.size.size());
    _pcm_sample_count = in.size[0];
    _pcm_buffer->resize(_pcm_sample_count);
    std::memcpy(_pcm_buffer->ptr(), in.input,
                sizeof(short) * _pcm_sample_count);

    // frame_size must be more than _feat_frame_count
    int frame_size = (_pcm_sample_count - _pcm_clip_duration_samples) /
                         _pcm_clip_stride_samples +
                     1;
    DBLOG << "_pcm_sample_count = " << _pcm_sample_count;
    DBLOG << "frame_size = " << frame_size;

    int pcm_size_in_bytes = _pcm_sample_count * sizeof(short);
    int feat_buffer_size = (_feat_ext_frame_dim * frame_size) * 3;
    const std::unique_ptr<Buffer<float>>& feat_buffer = get_buffer(_in_seq);
    feat_buffer->resize(feat_buffer_size);

    int feat_buffer_size_in_bytes = feat_buffer_size * sizeof(float);
    _feat_ext->SetBuff((void*)(_pcm_buffer->ptr()), pcm_size_in_bytes,
                       (void*)(feat_buffer->ptr()), feat_buffer_size_in_bytes);
    _feat_ext->Reset();

    BCHECK_LT(pcm_size_in_bytes, _feat_ext_package_size)
        << "Error, pcm_size_in_bytes is more than _feat_ext_package_size, "
        << "_pcm_size_in_bytes =  " << pcm_size_in_bytes
        << "_feat_ext_package_size = " << _feat_ext_package_size;

    // Extract features
    for (int remaining_bytes = pcm_size_in_bytes; remaining_bytes > 0;
         remaining_bytes -= _feat_ext_package_size) {
      int size_fed = _feat_ext_package_size < remaining_bytes
                         ? _feat_ext_package_size
                         : remaining_bytes;
      _feat_ext->ExtractFeat((char*)(_pcm_buffer->ptr()) /*+bytes?*/, size_fed,
                             &_feat_frame_count, 0);  // ?
    }

    _feat_ext->ExtractFeat(NULL, 0, &_feat_frame_count, 1);

    BCHECK_GT(_feat_frame_count, 0)
        << "_feat_frame_count is " << _feat_frame_count;
    DBLOG << "_feat_frame_count = " << _feat_frame_count;

    // Apply CMVN
    float* feat_ptr = feat_buffer->ptr();
    float* mean_ptr = _cmvn_mean_array->ptr();
    float* std_ptr = _cmvn_std_array->ptr();
    for (int frame_idx = 0; frame_idx < _feat_frame_count; frame_idx++) {
      for (int dim_idx = 0; dim_idx < _feat_ext_frame_dim; dim_idx++) {
        long long index = _feat_ext_frame_dim * frame_idx + dim_idx;
        float v = feat_ptr[index];
        // Calculate sqrt and reciprocal prior to this once and for all.
        v = (v + mean_ptr[dim_idx]) * std_ptr[dim_idx];
        feat_ptr[index] = v;
      }
    }

    // shape
    const long long batch_size = 1;
    std::vector<std::size_t>& shape_seq = get_shape(_in_seq);
    shape_seq.clear();
    shape_seq.push_back(_feat_frame_count);
    shape_seq.push_back(batch_size);
    shape_seq.push_back(_feat_ext_frame_dim);

    // seq_len
    const std::unique_ptr<Buffer<int>>& seq_len_buffer = get_buffer(_in_len);
    seq_len_buffer->resize(1);
    (seq_len_buffer->ptr())[0] = _feat_frame_count;

    std::vector<std::size_t>& shape_len =
        get_shape(_in_len);  // must be reference
    shape_len.clear();
    shape_len.push_back(batch_size);

    // set input and output
    BCHECK_EQ(0, set_input(_in_seq)) << "set input _in_seq error";
    BCHECK_EQ(0, set_input(_in_len)) << "set input _in_len error";
    BCHECK_EQ(0, set_output(_out)) << "set output error";

    return 0;
  }

  int post_process(Output& out) override {
    DBLOG << "AttentionOp post_process";
    // TODO ...
    BCHECK_EQ(0, get_result(_out)) << "Error, get_restul";

    std::vector<std::size_t>& shape = get_shape(_out);
    BCHECK(shape.size());
    int num_elem = shape[0];
    DBLOG << "out shape is [" << num_elem << "]";

    const std::unique_ptr<Buffer<int>>& result_buffer = get_buffer(_out);
    BCHECK_GE(result_buffer->size(), num_elem);
    BCHECK(_char_list.size()) << "_char_list size is " << _char_list.size();
    index_to_char(result_buffer->ptr(), num_elem);

    bool& confidence = out.confidence;
    confidence = true;
    (out.results).push_back(_inference_result);

    return 0;
  }

  DATA_TYPE<float> _in_seq;
  DATA_TYPE<int> _in_len;
  DATA_TYPE<int> _out;

  std::unique_ptr<Buffer<short>> _pcm_buffer;
  std::unique_ptr<Buffer<float>> _feat_buffer;

  std::size_t _pcm_sample_count;

  // mean and std
  std::unique_ptr<Buffer<float>> _cmvn_mean_array;
  std::unique_ptr<Buffer<float>> _cmvn_std_array;

  std::unique_ptr<Frontend> _feat_ext;
  int _feat_ext_sample_rate;
  int _feat_ext_package_size;
  int _feat_ext_frame_dim;
  int _feat_frame_count;

  std::vector<std::pair<int, std::string>> _char_list;  //<index, char>

  // result
  std::string _inference_result;

 private:
  // helper private function
  void init_feat_ext() {
    BCHECK(!_feat_ext) << "Error, _feat_ext is not nullptr";
    _feat_ext = make_unique<Frontend>();
    BCHECK(_feat_ext) << "Failed to create Frontend.";

    const std::string conf_file = _config_path + "/fbank.cfg";
    BLOG << "Using FE conf file: " << conf_file;
    BCHECK_NE(-1, _feat_ext->Init(const_cast<char*>(conf_file.c_str())))
        << "Failed to init Frontend." << std::endl;

    _feat_ext_sample_rate = _feat_ext->GetSampleRate();
    _feat_ext_package_size = (_feat_ext->GetFeatConf()->package_size);
    _feat_ext_frame_dim = _feat_ext->GetFeatureDim();

    BLOG << "_feat_ext_sample_rate: " << _feat_ext_sample_rate;
    BLOG << "_feat_ext_package_size: " << _feat_ext_package_size;
    BLOG << "_feat_ext_frame_dim: " << _feat_ext_frame_dim;

#if 0
        // Allocate various buffers.
        const float max_time_length = 200;  // seconds. TODO: hard-coded
        _pcm_buffer_size = _feat_ext_sample_rate * max_time_length + 200;
        _pcm_buffer = new short[_pcm_buffer_size];
        _feat_buffer_size = 100 * max_time_length;
        _feat_buffer = new float[_feat_ext_frame_dim * _feat_buffer_size];

        // Set feature extractor buffer size.
        _feat_ext->SetBuff((void*)_pcm_buffer, _pcm_buffer_size * 2,
            (void*)_feat_buffer, 10000 * _feat_ext_frame_dim * 4);
        _feat_ext->Reset();
#endif
  }

  // may be move to base Operator is better...
  void init_cmvn_array() {
    // Hard-coded CMVN file.
    const std::string cmvn_path = _config_path + "/global.fbank40_std.cmvn";
    BLOG << "Loading CMVN from file: " << cmvn_path << std::endl;

    BCHECK(_feat_ext_frame_dim);
    int cmvn_size = _feat_ext_frame_dim;
    _cmvn_mean_array = make_unique<Buffer<float>>(cmvn_size);
    _cmvn_std_array = make_unique<Buffer<float>>(cmvn_size);

    std::ifstream cmvn_ifs(cmvn_path);
    BCHECK(cmvn_ifs) << "Failed open cmvn file : " << cmvn_path;

    int read_size = 0;
    float cmvn_value = 0.0f;
    while (!cmvn_ifs.eof()) {
      cmvn_ifs >> cmvn_value;
      if (cmvn_ifs.eof() || read_size >= cmvn_size * 2) {
        break;
      }
      if (read_size % 2 == 0) {
        (_cmvn_mean_array->ptr())[read_size / 2] = -cmvn_value;
      } else {
        (_cmvn_std_array->ptr())[read_size / 2] = 1 / sqrt(cmvn_value);
      }
      read_size++;
    }

    BLOG << "Loaded " << read_size << " cmvn values.";
  }

  // shoud be move to base Operator ...
  void init_char_list() {
    const std::string attention_char_file =
        _config_path + "/chars5003_attention.txt";
    DBLOG << "Loading char list from file: " << attention_char_file;
    std::ifstream char_list_ifs(attention_char_file);
    BCHECK(char_list_ifs) << "failed open char file : " << attention_char_file;

    while (true) {
      if (char_list_ifs.eof()) {
        break;
      }
      std::string char_txt;
      std::string char_idx;
      char_list_ifs >> char_txt;
      char_list_ifs >> char_idx;
      int index = atoi(char_idx.c_str());
      _char_list.push_back(std::make_pair(index, char_txt));
    }

    // for (int i = 0; i < _char_list.size(); ++i) {
    //    DBLOG << "i = " << i << ", " << _char_list[i].second;
    //}
    char_list_ifs.close();
  }

  void index_to_char(int* index_seq, long long seq_size) {
    std::vector<std::string> result;
    for (int i = 0; i < seq_size; ++i) {
      if (index_seq[i] != 5005 && index_seq[i] != 5004) {
        // DBLOG << "index = " << index_seq[i];
        // result.push_back(_char_list[index_seq[i] - 1].second);
        result.push_back(_char_list[index_seq[i]].second);
      }
    }

    std::stringstream ss;
    long size = result.size();
    for (long long ii = 0; ii < size; ++ii) {
      // delete "null" or "ring"
      if ("n" == result[ii] || "r" == result[ii]) {
        if (ii + 3 < size) {
          std::stringstream null_ring_s;
          null_ring_s << result[ii];
          null_ring_s << result[ii + 1];
          null_ring_s << result[ii + 2];
          null_ring_s << result[ii + 3];
          if ("null" == null_ring_s.str() || "ring" == null_ring_s.str()) {
            ii += 4;
            continue;
          }
        }
      }

      if (result[ii] != "sil" && result[ii] != "START" && result[ii] != "END") {
        ss << result[ii];
      }
    }
    _inference_result = ss.str();
  }
};

RESISTER_OPERATOR(attention, AttentionOp);

}  // namespace operators

}  // namespace inference
