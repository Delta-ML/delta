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
 * @file emotion_speech_with_text_op.cc
 * @brief This emotion model train by heyahao and zhanghui
 * @author gaoyonghu
 * @version 0.1
 * @date 2019-01-17
 *
 * SplitWord https://www.cnblogs.com/chinxi/p/6129774.html
 * utf-8是一种变长编码
 * 1）对于单字节的符号，字节的第一位设为0，后面7位为这个符号的unicode码。
 *    因此对于英语字母，UTF-8编码和ASCII码是相同的。
 * 2）对于n字节的符号（n>1），第一个字节的前n位都设为1，第n+1位设为0，后面字节的前两位一律设为10。
 *    剩下的没有提及的二进制位，全部为这个符号的unicode码。
 * 如表：
 *  1字节    0xxxxxxx
 *  2字节    110xxxxx 10xxxxxx
 *  3字节    1110xxxx 10xxxxxx 10xxxxxx
 *  4字节    11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
 *  5字节    111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
 *  6字节    1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
 * */

#include <fstream>
#include "base.h"
#include "dynload/feature_extraction_wrapper.h"
#include "operator.h"

namespace inference {

namespace operators {

using namespace FeatureExtraction;

namespace {
// TODO ...
void splitWord(const std::string& word, std::vector<std::string>& characters) {
  characters.clear();
  int num = word.size();
  int i = 0;
  while (i < num) {
    int size = 1;
    if (word[i] & 0x80) {
      char temp = word[i];
      temp <<= 1;
      do {
        temp <<= 1;
        ++size;
      } while (temp & 0x80);
    }
    std::string subWord;
    subWord = word.substr(i, size);
    characters.push_back(subWord);
    i += size;
  }
}

void word_to_id(const std::vector<std::string>& characters,
                const std::unordered_map<std::string, int> word_list,
                std::unique_ptr<Buffer<int>>& id_buffer) {
  BCHECK(characters.size());
  BCHECK(id_buffer);
  BCHECK(word_list.size());

  // id_buffer->resize(characters.size());
  int* ptr = id_buffer->ptr();
  int word_num = characters.size();
  for (int i = 0; i < word_num; ++i) {
    auto it = word_list.find(characters[i]);
    ptr[i] = it->second;
  }
}

}  // end namespace

class EmotionSpeechWithTextOp : public Operator {
 public:
  EmotionSpeechWithTextOp(const OperatorConfig& config) : Operator(config) {
    BLOG << "EmotionSpeechWithTextOp Construct, _threshold is " << _threshold;

    std::vector<std::size_t> shape;
    _speech_in = make_tuple("input", make_unique<Buffer<float>>(), shape);
    _text_in = make_tuple("input_text", make_unique<Buffer<int>>(), shape);
    _out = make_tuple("softmax_output", make_unique<Buffer<float>>(), shape);

    float window = _window / _scalar;  // 0.025s
    float step = _step / _scalar;      // 0.01s
    bool delta = true;
    _fbank =
        FBANK_CREATE(window, step, _sample_rate, _filter_num, delta);  // init

    // init word dict
    init_word_list();
  }

  ~EmotionSpeechWithTextOp() {
    if (!_fbank) {
      FBANK_DESTROY(_fbank);
      _fbank = nullptr;
    }
  }

 private:
  int pre_process(Input& in) override {
    DBLOG << "EmotionSpeechWithTextOp pre_process, this is " << this;
    // TODO ...
    BCHECK(in.input);
    BCHECK(in.input_text);
    BCHECK((in.input_text)->size());
    int char_size = in.input_text->size();

    splitWord(*(in.input_text), _characters);
    int word_size = _characters.size();
    for (int i = 0; i < word_size; ++i) {
      DBLOG << "i = " << i << ", " << _characters[i];
    }

    // word to id
    std::unique_ptr<Buffer<int>>& id_buffer = get_buffer(_text_in);
    int text_frame = (word_size + (_text_dim - 1)) / _text_dim;
    id_buffer->resize(text_frame * _text_dim);
    id_buffer->zero();
    word_to_id(_characters, _word_list, id_buffer);
    BCHECK(id_buffer->size());
    for (int i = 0; i < word_size; ++i) {
      BLOG << "word : " << _characters[i] << ", ID : " << (id_buffer->ptr())[i];
    }

    BLOG << "text shape (" << text_frame << ", " << _text_dim << ")";
    std::vector<std::size_t>& txt_shape =
        get_shape(_text_in);  // must be reference
    txt_shape.clear();
    txt_shape.push_back(text_frame);
    txt_shape.push_back(_text_dim);

    // feature extraction
    int pcm_sample_count = in.size[0];
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

    std::unique_ptr<Buffer<float>>& buffer = get_buffer(_speech_in);

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

    BLOG << "speech shape (" << batch_num << ", " << _batch << ", "
         << _filter_num << ", " << _channel << ")";

    std::vector<std::size_t>& shape =
        get_shape(_speech_in);  // must be reference
    shape.clear();
    shape.push_back(batch_num);
    shape.push_back(_batch);
    shape.push_back(_filter_num);
    shape.push_back(_channel);

    // set input and output
    BCHECK_EQ(0, set_input(_speech_in)) << "set input speech error";
    BCHECK_EQ(0, set_input(_text_in)) << "set input text error";
    BCHECK_EQ(0, set_output(_out)) << "set output error";

    return 0;
  }

  int post_process(Output& out) override {
    DBLOG << "EmotionSpeechWithTextOp post_process";
    // TODO ...
    BCHECK_EQ(0, get_result(_out));
    std::vector<std::size_t>& shape = get_shape(_out);  // must be reference
    BCHECK(shape.size()) << "out shape is " << shape.size();
    int batch_size = shape[0];
    int dim = shape[1];
    BLOG << "out shape is (" << batch_size << ", " << dim << ")";

    std::vector<float> score_result;
    float* ptr = get_buffer(_out)->ptr();
    float sum = 0.0f;
    for (int n = 0; n < batch_size; ++n) {
      BLOG << "[" << *ptr << ", " << *(ptr + 1) << "]";
      score_result.push_back(*(ptr + 1));
      sum += *(ptr + 1);
      ptr += dim;
    }
    sum /= batch_size;
    DBLOG << "score_result size is " << score_result.size();

    out.confidence = sum > _threshold ? true : false;

    std::stringstream ss;

    ss << "{ \"model_type\":\"emotion_speech_with_text\", ";  // emotion_type
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

 private:
  DATA_TYPE<float> _speech_in;
  DATA_TYPE<int> _text_in;
  DATA_TYPE<float> _out;

  FilterBank* _fbank = nullptr;
  int _filter_num = 40;
  int _channel = 3;
  int _batch = 2000;
  int _text_dim = 100;

  std::vector<std::string> _characters;
  std::unordered_map<std::string, int> _word_list;  //<word, index>

  // shoud be move to base Operator ...
  void init_word_list() {
    const std::string emotion_char_file = _config_path + "/chars5048.txt";
    DBLOG << "Loading word dict from file: " << emotion_char_file;
    std::ifstream word_list_ifs(emotion_char_file.c_str());
    BCHECK(word_list_ifs) << "failed open word dict file : "
                          << emotion_char_file;

    int index = 0;
    while (true) {
      if (word_list_ifs.eof()) {
        break;
      }
      std::string char_txt;
      word_list_ifs >> char_txt;
      _word_list.insert({char_txt, index});
      ++index;
    }

    BLOG << "word dict size is " << _word_list.size();
    // for(auto& e : _word_list) {
    //    BLOG << "(" << e.first << ", " << e.second << ")";
    //}
    word_list_ifs.close();
  }
};

RESISTER_OPERATOR(emotion_speech_with_text, EmotionSpeechWithTextOp);

}  // namespace operators

}  // namespace inference
