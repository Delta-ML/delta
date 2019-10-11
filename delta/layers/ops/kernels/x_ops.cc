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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

// See tensorflow/core/ops/audio_ops.cc
namespace delta {
namespace {

using namespace tensorflow;  // NOLINT
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::UnchangedShape;

Status PitchShapeFn(InferenceContext* c) {
  ShapeHandle input_data;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_data));

  int wav_len = c->Value(c->Dim(input_data, 0));
  float SampleRate = 16000.0f;
  float win_len, frm_len;
  TF_RETURN_IF_ERROR(c->GetAttr("window_length", &win_len));
  TF_RETURN_IF_ERROR(c->GetAttr("frame_length", &frm_len));
  int i_WinLen = static_cast<int>(win_len * SampleRate);
  int i_FrmLen = static_cast<int>(frm_len * SampleRate);
  int i_NumFrm = (wav_len - i_WinLen) / i_FrmLen + 1;
  c->set_output(0, c->Vector(i_NumFrm));

  return Status::OK();
}

Status FrmPowShapeFn(InferenceContext* c) {
  ShapeHandle input_data;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_data));

  int wav_len = c->Value(c->Dim(input_data, 0));
  float SampleRate = 16000.0f;
  float win_len, frm_len;
  TF_RETURN_IF_ERROR(c->GetAttr("window_length", &win_len));
  TF_RETURN_IF_ERROR(c->GetAttr("frame_length", &frm_len));
  int i_WinLen = static_cast<int>(win_len * SampleRate);
  int i_FrmLen = static_cast<int>(frm_len * SampleRate);
  int i_NumFrm = (wav_len - i_WinLen) / i_FrmLen + 1;
  c->set_output(0, c->Vector(i_NumFrm));

  return Status::OK();
}

Status ZcrShapeFn(InferenceContext* c) {
  ShapeHandle input_data;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_data));

  int wav_len = c->Value(c->Dim(input_data, 0));
  float SampleRate = 16000.0f;
  float win_len, frm_len;
  TF_RETURN_IF_ERROR(c->GetAttr("window_length", &win_len));
  TF_RETURN_IF_ERROR(c->GetAttr("frame_length", &frm_len));
  int i_WinLen = static_cast<int>(win_len * SampleRate);
  int i_FrmLen = static_cast<int>(frm_len * SampleRate);
  int i_NumFrm = (wav_len - i_WinLen) / i_FrmLen + 1;
  c->set_output(0, c->Vector(i_NumFrm));

  return Status::OK();
}

Status SpectrumShapeFn(InferenceContext* c) {
  ShapeHandle input_data;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_data));

  int wav_len = c->Value(c->Dim(input_data, 0));
  float SampleRate = 16000.0f;
  float win_len, frm_len;
  TF_RETURN_IF_ERROR(c->GetAttr("window_length", &win_len));
  TF_RETURN_IF_ERROR(c->GetAttr("frame_length", &frm_len));
  int i_WinLen = static_cast<int>(win_len * SampleRate);
  int i_FrmLen = static_cast<int>(frm_len * SampleRate);
  int i_NumFrm = (wav_len - i_WinLen) / i_FrmLen + 1;
  int i_FrqNum = static_cast<int>(pow(2.0f, ceil(log2(i_WinLen))) / 2 + 1);
  c->set_output(0, c->Matrix(i_NumFrm, i_FrqNum));

  return Status::OK();
}

Status CepstrumShapeFn(InferenceContext* c) {
  ShapeHandle input_data;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_data));

  int wav_len = c->Value(c->Dim(input_data, 0));
  float SampleRate = 16000.0f;
  float win_len, frm_len;
  int ceps_sub_num;
  TF_RETURN_IF_ERROR(c->GetAttr("window_length", &win_len));
  TF_RETURN_IF_ERROR(c->GetAttr("frame_length", &frm_len));
  TF_RETURN_IF_ERROR(c->GetAttr("ceps_subband_num", &ceps_sub_num));

  int i_WinLen = static_cast<int>(win_len * SampleRate);
  int i_FrmLen = static_cast<int>(frm_len * SampleRate);
  int i_NumFrm = (wav_len - i_WinLen) / i_FrmLen + 1;
  c->set_output(0, c->Matrix(i_NumFrm, ceps_sub_num));

  return Status::OK();
}

Status PLPShapeFn(InferenceContext* c) {
  ShapeHandle input_data;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_data));

  int wav_len = c->Value(c->Dim(input_data, 0));
  float SampleRate = 16000.0f;
  float win_len, frm_len;
  int plp_order;
  TF_RETURN_IF_ERROR(c->GetAttr("window_length", &win_len));
  TF_RETURN_IF_ERROR(c->GetAttr("frame_length", &frm_len));
  TF_RETURN_IF_ERROR(c->GetAttr("plp_order", &plp_order));
  int i_WinLen = static_cast<int>(win_len * SampleRate);
  int i_FrmLen = static_cast<int>(frm_len * SampleRate);
  int i_NumFrm = (wav_len - i_WinLen) / i_FrmLen + 1;
  c->set_output(0, c->Matrix(i_NumFrm, plp_order + 1));

  return Status::OK();
}

Status AnalyfiltbankShapeFn(InferenceContext* c) {
  ShapeHandle input_data;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_data));

  int wav_len = c->Value(c->Dim(input_data, 0));
  float SampleRate = 16000.0f;
  float win_len, frm_len;
  TF_RETURN_IF_ERROR(c->GetAttr("window_length", &win_len));
  TF_RETURN_IF_ERROR(c->GetAttr("frame_length", &frm_len));
  int i_WinLen = static_cast<int>(win_len * SampleRate);
  int i_FrmLen = static_cast<int>(frm_len * SampleRate);
  int i_NumFrm = (wav_len - i_WinLen) / i_FrmLen + 1;
  int i_FrqNum = static_cast<int>(pow(2.0f, ceil(log2(i_WinLen))) / 2 + 1);
  c->set_output(0, c->Matrix(i_NumFrm, i_FrqNum));
  c->set_output(1, c->Matrix(i_NumFrm, i_FrqNum));

  return Status::OK();
}

Status SynthfiltbankShapeFn(InferenceContext* c) {
  ShapeHandle input_data_1;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_data_1));
  ShapeHandle input_data_2;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input_data_2));

  int i_NumFrm = c->Value(c->Dim(input_data_1, 0));
  float SampleRate = 16000.0f;
  float win_len, frm_len;
  TF_RETURN_IF_ERROR(c->GetAttr("window_length", &win_len));
  TF_RETURN_IF_ERROR(c->GetAttr("frame_length", &frm_len));
  int i_WinLen = static_cast<int>(win_len * SampleRate);
  int i_FrmLen = static_cast<int>(frm_len * SampleRate);
  int L = (i_NumFrm - 1) * i_FrmLen + i_WinLen;
  c->set_output(0, c->Vector(L));

  return Status::OK();
}

Status FbankShapeFn(InferenceContext* c) {
  ShapeHandle spectrogram;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &spectrogram));
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

  int32 filterbank_channel_count;
  TF_RETURN_IF_ERROR(
      c->GetAttr("filterbank_channel_count", &filterbank_channel_count));

  DimensionHandle audio_channels = c->Dim(spectrogram, 0);
  DimensionHandle spectrogram_length = c->Dim(spectrogram, 1);

  DimensionHandle output_channels = c->MakeDim(filterbank_channel_count);

  c->set_output(
      0, c->MakeShape({audio_channels, spectrogram_length, output_channels}));
  return Status::OK();
}

Status NgramShapeFn(InferenceContext* c) {
  int word_ngrams = 2;
  TF_RETURN_IF_ERROR(c->GetAttr("word_ngrams", &word_ngrams));

  ShapeHandle sent = c->input(0);
  int32 rank = c->Rank(sent);
  int32 input_len;
  ShapeHandle batch_dims;
  if (rank > 1) {
    TF_RETURN_IF_ERROR(c->Subshape(sent, 0, -1, &batch_dims));
    input_len = c->Value(c->Dim(sent, -1));
  } else {
    DimensionHandle sent_len = c->Dim(sent, 0);
    input_len = c->Value(sent_len);
  }

  int output_len;
  if (input_len <= 0) {
    output_len = 0;
  } else if (word_ngrams <= 1) {
    output_len = input_len;
  } else {
    int x = input_len > word_ngrams ? input_len - word_ngrams + 1 : 1;
    int l = input_len - x + 1;
    output_len = ((input_len + x) * l) >> 1;
  }

  if (rank > 1) {
    ShapeHandle ngram_dims = c->Vector(c->MakeDim(output_len));
    ShapeHandle out_shape;
    TF_RETURN_IF_ERROR(c->Concatenate(batch_dims, ngram_dims, &out_shape));
    c->set_output(0, out_shape);
  } else {
    DimensionHandle ngram_dim = c->MakeDim(output_len);
    c->set_output(0, c->Vector(ngram_dim));
  }

  return Status::OK();
}

Status SentenceToIdsShapeFn(InferenceContext* c) {
  ShapeHandle sent = c->input(0);
  int32 rank = c->Rank(sent);
  if (rank > 0) {
    auto batch_size = c->Dim(c->input(0), 0);
    int maxlen;
    TF_RETURN_IF_ERROR(c->GetAttr("maxlen", &maxlen));
    c->set_output(0, c->Matrix(batch_size, maxlen));
    c->set_output(1, c->Matrix(batch_size, maxlen));
  } else {
    int maxlen;
    TF_RETURN_IF_ERROR(c->GetAttr("maxlen", &maxlen));
    c->set_output(0, c->Vector(maxlen));
    c->set_output(1, c->Vector(maxlen));
  }

  return Status::OK();
}

}  // namespace

// TF:
// wav to spectrogram:
// https://github.com/tensorflow/tensorflow/examples/wav_to_spectrogram/wav_to_spectrogram.h
// audio ops: https://github.com/tensorflow/tensorflow/core/ops/audio_ops.cc
// DecodeWav:
// tensorflow/tensorflow/core/api_def/base_api/api_def_DecodeWav.pbtxt
// EncodeWav:
// tensorflow/tensorflow/core/api_def/base_api/api_def_EncodeWav.pbtxt
// AudioSpectrogram:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/spectrogram_op.cc
//     tensorflow/tensorflow/core/api_def/base_api/api_def_AudioSpectrogram.pbtxt
// Mfcc:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/mfcc_op.cc
//     tensorflow/tensorflow/core/api_def/base_api/api_def_Mfcc.pbtxt

// TFLite
// mfcc:
// https://github.com/tensorflow/tensorflow/tensorflow/lite/kernels/internal/mfcc_mel_filterbank.h
// spectorgram: tensorflow/tensorflow/lite/kernels/internal/spectrogram.h

REGISTER_OP("Pitch")
    .Input("input_data: float")
    .Input("sample_rate: float")
    .Attr("window_length: float = 0.025")
    .Attr("frame_length: float = 0.010")
    .Attr("thres_autoc: float = 0.3")
    .Output("output: float")
    .SetShapeFn(PitchShapeFn)
    .Doc(R"doc(
    Create pitch feature files.
    input_data: float, input wave, a tensor of shape [1, data_length].
    sample_rate: float, NB 8000, WB 16000 etc.
    window_length: float, window length in second.
    frame_length: float, frame length in second. 
    output: float, pitch features, [num_Frame].
    )doc");

REGISTER_OP("FramePow")
    .Input("input_data: float")
    .Input("sample_rate: float")
    .Attr("window_length: float = 0.025")
    .Attr("frame_length: float = 0.010")
    .Output("output: float")
    .SetShapeFn(FrmPowShapeFn)
    .Doc(R"doc(
    Create frame energy feature files.
    input_data: float, input wave, a tensor of shape [1, data_length].
    sample_rate: float, NB 8000, WB 16000 etc.
    window_length: float, window length in second.
    frame_length: float, frame length in second. 
    output: float, frame energy features, [num_Frame].
    )doc");

REGISTER_OP("ZCR")
    .Input("input_data: float")
    .Input("sample_rate: float")
    .Attr("window_length: float = 0.025")
    .Attr("frame_length: float = 0.010")
    .Output("output: float")
    .SetShapeFn(ZcrShapeFn)
    .Doc(R"doc(
    Create zero cross rate feature files.
    input_data: float, input wave, a tensor of shape [1, data_length].
    sample_rate: float, NB 8000, WB 16000 etc.
    window_length: float, window length in second.
    frame_length: float, frame length in second. 
    output: float, zero cross rate features, [num_Frame].
    )doc");

REGISTER_OP("Spectrum")
    .Input("input_data: float")
    .Input("sample_rate: float")
    .Attr("window_length: float = 0.025")
    .Attr("frame_length: float = 0.010")
    .Attr("output_type: int = 2")
    .Output("output: float")
    .SetShapeFn(SpectrumShapeFn)
    .Doc(R"doc(
    Create spectrum feature files.
    input_data: float, input wave, a tensor of shape [1, data_length].
    sample_rate: float, NB 8000, WB 16000 etc.
    window_length: float, window length in second.
    frame_length: float, frame length in second.
    output_type: int, 1: PSD, 2: log(PSD) 
    output: float, PSD/logPSD features, [num_Frame, num_Subband].
    )doc");

REGISTER_OP("Cepstrum")
    .Input("input_data: float")
    .Input("sample_rate: float")
    .Attr("window_length: float = 0.025")
    .Attr("frame_length: float = 0.010")
    .Attr("ceps_subband_num: int = 13")
    .Attr("tag_ceps_mean_norm: bool = true")
    .Output("output: float")
    .SetShapeFn(CepstrumShapeFn)
    .Doc(R"doc(
    Create cepstrum feature files.
    input_data: float, input wave, a tensor of shape [1, data_length].
    sample_rate: float, NB 8000, WB 16000 etc.
    window_length: float, window length in second.
    frame_length: float, frame length in second. 
    output: float, cepstrum features, [num_Frame, num_Ceps].
    )doc");

REGISTER_OP("PLP")
    .Input("input_data: float")
    .Input("sample_rate: float")
    .Attr("window_length: float = 0.025")
    .Attr("frame_length: float = 0.010")
    .Attr("plp_order: int = 12")
    .Output("output: float")
    .SetShapeFn(PLPShapeFn)
    .Doc(R"doc(
    Create plp feature files.
    input_data: float, input wave, a tensor of shape [1, data_length].
    sample_rate: float, NB 8000, WB 16000 etc.
    window_length: float, window length in second.
    frame_length: float, frame length in second. 
    output: float, plp features, [num_Frame, plp_order+1].
    )doc");

REGISTER_OP("Analyfiltbank")
    .Input("input_data: float")
    .Input("sample_rate: float")
    .Attr("window_length: float = 0.030")
    .Attr("frame_length: float = 0.010")
    .Output("output_1: float")
    .Output("output_2: float")
    .SetShapeFn(AnalyfiltbankShapeFn)
    .Doc(R"doc(
    Create analysis filter bank feature files.
    input_data: float, input wave, a tensor of shape [1, data_length].
    sample_rate: float, NB 8000, WB 16000 etc.
    window_length: float, window length in second, support 30ms only for now.
    frame_length: float, frame length in second, support 10ms only for now.
    output_1: float, power spectrum, [num_Frame, num_Subband]
    output_2: float, phase spectrum, [num_Frame, num_Subband]
    )doc");

REGISTER_OP("Synthfiltbank")
    .Input("input_data_1: float")
    .Input("input_data_2: float")
    .Input("sample_rate: float")
    .Attr("window_length: float = 0.030")
    .Attr("frame_length: float = 0.010")
    .Output("output: float")
    .SetShapeFn(SynthfiltbankShapeFn)
    .Doc(R"doc(
    Create synthesis filter bank feature files.
    input_data_1: float, power spectrum, a tensor of shape [num_frm, num_freq].
    input_data_2: float, phase spectrum, a tensor of shape [num_frm, num_freq].
    sample_rate: float, NB 8000, WB 16000 etc.
    window_length: float, window length in second, support 30ms only for now.
    frame_length: float, frame length in second, support 10ms only for now.
    output: float, reconstructed wave file, [L].
    )doc");

REGISTER_OP("Fbank")
    .Input("spectrogram: float")
    .Input("sample_rate: int32")
    .Attr("upper_frequency_limit: float = 4000")
    .Attr("lower_frequency_limit: float = 20")
    .Attr("filterbank_channel_count: int = 40")
    .Output("output: float")
    .SetShapeFn(FbankShapeFn)
    .Doc(R"doc(
Create Mel-filter bank (FBANK) feature files.
spectrogram
: float, A tensor of shape [audio_channels, spectrogram_length, spectrogram_feat_dim].
sample_rate: int32, how many samples per second the source audio used. e.g. 16000, 8000.
upper_frequency_limit: float, the highest frequency to use when calculating the ceptstrum.
lower_frequency_limit: float, the lowest frequency to use when calculating the ceptstrum.
filterbank_channel_count: int, resolution of the Mel bank used internally. 
output: float, fbank features, a tensor of shape [audio_channels, spectrogram_length, bank_feat_dim].
)doc");

// ref: https//github.com/kaldi-asr/kaldi/src/featbin/add-deltas.cc
REGISTER_OP("DeltaDelta")
    .Input("features: float")
    .Output("features_with_delta_delta: float")
    .Attr("order: int = 2")
    .Attr("window: int = 2")
    .SetShapeFn([](InferenceContext* c) {
      int order;
      TF_RETURN_IF_ERROR(c->GetAttr("order", &order));

      ShapeHandle feat;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &feat));

      DimensionHandle nframe = c->Dim(feat, 0);
      DimensionHandle nfeat = c->Dim(feat, 1);

      DimensionHandle out_feat;
      TF_RETURN_IF_ERROR(c->Multiply(nfeat, order + 1, &out_feat));

      // int input_len = c->Value(nfeat);
      // int output_len = input_len * (order + 1);
      // DimensionHandle out_feat = c->MakeDim(output_len);
      c->set_output(0, c->Matrix(nframe, out_feat));
      return Status::OK();
    })
    .Doc(R"doc(
Add deltas (typically to raw mfcc or plp features).
features: A matrix of shape [nframe, feat_dim].
features_with_delta_delta: A matrix of shape [nframe, feat_dim * (order + 1)].
order: int, order fo delta computation.
window: a int, parameter controlling window for delta computation(actual window
    size for each delta order is 1 + 2*window).
)doc");

// Usage:
//     ngram_out = ngram(sentence_in, vocab_size, bucket_size, word_ngrams)
// Input:
//     sentence_in: the input sentence, a list with the index of each word
//     vocab_size: vocab size
//     bucket_size: ngram bucket size
//     word_ngrams: 1-gram, 2-gram, 3-gram,...
// Output:
//     ngram_out: unigram idx + ngram idx

REGISTER_OP("Ngram")
    .Input("sentence_in: T")
    .Output("ngram_out: T")
    .Attr("word_ngrams: int = 2")
    .Attr("vocab_size: int >= 1")
    .Attr("bucket_size: int >= 1")
    .Attr("T: {int32, int64}")
    .SetShapeFn(NgramShapeFn)
    .Doc(R"doc(
    Calculate the ngram of the input sentence, both the input sentence and output
    ngram are represented by word index.
    input: int, input sentence. A tensor of shape [batch_size, sentence_size] or [sentence_size].
    vocab_size: int, vocab size.
    bucket_size: int, ngram bucket size.
    word_ngrams: int, the value of n.
    output: int,  A tensor of shape [batch_size, ngram_size] or [ngram_size].
    )doc");

REGISTER_OP("VocabTokenToId")
    .Input("token: string")
    .Output("id: int32")
    .Attr("vocab: list(string)")
    .Attr("load_token_ids_from_vocab: bool = false")
    .SetIsStateful()
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Looks up the token in the vocab and return its id.
token: A scalar or list of strings.
id: A scalar or list of ints.
vocab: A list of strings.
load_token_ids_from_vocab: Whether token ids are present in vocab (i.e. vocab
    contains two colums, one for IDs and one for words).  If false, line numbers
    are used.
)doc");

REGISTER_OP("VocabIdToToken")
    .Input("id: int32")
    .Output("token: string")
    .Attr("vocab: list(string)")
    .Attr("load_token_ids_from_vocab: bool = false")
    .SetIsStateful()
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Looks up the token at the given id from a vocab.
id: A scalar or list of ints.
token: A scalar or list of strings.
vocab: A list of strings.
load_token_ids_from_vocab: Whether token ids are present in vocab (i.e. vocab
    contains two colums, one for IDs and one for words).  If false, line numbers
    are used.
)doc");

REGISTER_OP("TokenInVocab")
    .Input("token: string")
    .Output("result: bool")
    .Attr("vocab: list(string)")
    .Attr("load_token_ids_from_vocab: bool = false")
    .SetIsStateful()
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Checks whether the provided token is in the vocab.
token: A scalar or list of strings.
result: A scalar or list of bools.
vocab: A list of strings.
load_token_ids_from_vocab: Whether token ids are present in vocab (i.e. vocab
    contains two colums, one for IDs and one for words).  If false, line numbers
    are used.
)doc");

REGISTER_OP("JiebaCut")
    .Input("sentence_in: string")
    .Output("sentence_out: string")
    .Attr("use_file: bool = false")
    .Attr("dict_lines: list(string) = ['']")
    .Attr("model_lines: list(string) = ['']")
    .Attr("user_dict_lines: list(string) = ['']")
    .Attr("idf_lines: list(string) = ['']")
    .Attr("stop_word_lines: list(string) = ['']")
    .Attr("dict_path: string = ''")
    .Attr("hmm_path: string = ''")
    .Attr("user_dict_path: string = ''")
    .Attr("idf_path: string = ''")
    .Attr("stop_word_path: string = ''")
    .Attr("hmm: bool = true")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Cut the Chines sentence into words.
sentence_in: A scalar or list of strings.
sentence_out: A scalar or list of strings.
)doc");

REGISTER_OP("StrLower")
    .Input("input: string")
    .Output("output: string")
    .SetShapeFn(UnchangedShape);

REGISTER_OP("SentenceToIds")
    .Input("sentence_in: string")
    .Output("token_ids: int32")
    .Output("paddings: float")
    .Attr("maxlen: int = 300")
    .Attr("pad_to_maxlen: bool = true")
    .Attr("use_vocab_file: bool = false")
    .Attr("vocab: list(string) = ['']")
    .Attr("vocab_filepath: string = ''")
    .Attr("load_token_ids_from_vocab: bool = true")
    .Attr("delimiter: string = ' '")
    .Attr("pad_id: int = -1")
    .Attr("check_tokens: bool = true")
    .SetShapeFn(SentenceToIdsShapeFn)
    .Doc(R"doc(
Tokenizes string into white space separated tokens according to a vocab file.
sentence_in: A vector of shape [batch].
token_ids: A matrix of shape [batch, maxlen].
    token_ids[i, j] is the i-th sample's j-th token id.
    token_ids[i, 0] is always <s>.
paddings: A matrix of shape [batch, maxlen].
    paddings[i, j] == 1.0 indicates that i-th training example's
    j-th target token is padded and should be ignored.
maxlen: an integer, sequence length of the output tensors.
pad_to_maxlen: Whether to pad the output to maxlen.
vocab_filepath: a string, filepath to the vocab file.
load_token_ids_from_vocab: Whether token ids are present in vocab (i.e. vocab
    contains two colums, one for IDs and one for words).  If false, line numbers
    are used.
delimiter: The delimiter to split the labels to tokens by.
pad_id: The padding id.
check_tokens: If tokens like <unk> must be in vocab file.
)doc");

}  // namespace delta
