/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include <algorithm>      // NOLINT
#include <string>         // NOLINT
#include <unordered_map>  // NOLINT
#include <vector>         // NOLINT

#include "../../../kernels/simple_vocab.h"  //NOLINT
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;  // NOLINT

namespace delta {
namespace {

class SentenceToIdsOp : public OpKernel {
 public:
  explicit SentenceToIdsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("maxlen", &maxlen_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pad_to_maxlen", &pad_to_maxlen_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pad_id", &pad_id_));
    bool load_token_ids_from_vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("load_token_ids_from_vocab",
                                     &load_token_ids_from_vocab));

    bool use_vocab_file;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_vocab_file", &use_vocab_file));
    bool check_tokens;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("check_tokens", &check_tokens));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("delimiter", &delimiter_));
    CHECK_GT(maxlen_, 0);

    if (use_vocab_file) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_filepath", &vocab_filepath_));
      OP_REQUIRES_OK(ctx, vocab_.Load(vocab_filepath_,
                                      load_token_ids_from_vocab, check_tokens));
    } else {
      std::vector<string> vocab;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab", &vocab));
      OP_REQUIRES_OK(
          ctx, vocab_.Load(vocab, load_token_ids_from_vocab, check_tokens));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* t_sentence_in;
    OP_REQUIRES_OK(ctx, ctx->input("sentence_in", &t_sentence_in));

    if (t_sentence_in->dims() == 0) {
      const auto& sentence_in = t_sentence_in->scalar<tstring>()();
      Tensor token_ids(DT_INT32, TensorShape({maxlen_}));
      Tensor paddings(DT_FLOAT, TensorShape({maxlen_}));
      auto t_token_ids = token_ids.tensor<int32, 1>();
      auto t_paddings = paddings.tensor<float, 1>();
      t_token_ids.setZero();
      t_paddings.setZero();
      int actual_maxlen = pad_to_maxlen_ ? maxlen_ : 0;

      string sentence(sentence_in);
      VLOG(1) << "sentence " << sentence;
      std::vector<string> tokens;
      tokens =
          str_util::Split(sentence, delimiter_, str_util::SkipWhitespace());

      VLOG(1) << "#Tokens " << tokens.size() << " "
              << str_util::Join(tokens, "/");
      int cur_char = 0;
      for (const auto& token : tokens) {
        const int token_id = vocab_.TokenToId(token);
        // If the number of tokens is longer than the max length - truncate.
        if (cur_char >= maxlen_) {
          //          cur_char++;
          //          LOG(DEBUG) << "Sentence: \"" << sentence << "\" contained
          //          "
          //                    << tokens.size()
          //                    << " tokens, and was truncated to size: " <<
          //                    maxlen_ << " ("
          //                    << tokens.size() - maxlen_ << " tokens were
          //                    ignored).";
          break;
        }
        t_token_ids(cur_char) = token_id;
        t_paddings(cur_char) = 0.0;
        cur_char++;
      }
      actual_maxlen = std::max(actual_maxlen, cur_char);
      for (; cur_char < maxlen_; ++cur_char) {
        t_token_ids(cur_char) = pad_id_;
        t_paddings(cur_char) = 1.0;
      }

      Tensor out_token_ids(DT_INT32, TensorShape({actual_maxlen}));
      Tensor out_paddings(DT_FLOAT, TensorShape({actual_maxlen}));

      typedef const Eigen::DSizes<Eigen::DenseIndex, 1> DSize1;
      //      LOG(INFO) << "actual_maxlen: " << actual_maxlen << ", maxlen_: "
      //      << maxlen_;
      out_token_ids.vec<int32>() =
          t_token_ids.slice(DSize1{0}, DSize1{actual_maxlen});
      out_paddings.vec<float>() =
          t_paddings.slice(DSize1{0}, DSize1{actual_maxlen});
      OP_REQUIRES_OK(ctx, ctx->set_output("token_ids", out_token_ids));
      OP_REQUIRES_OK(ctx, ctx->set_output("paddings", out_paddings));
    } else {
      OP_REQUIRES(
          ctx, t_sentence_in->dims() == 1,
          errors::InvalidArgument("Input must be a scalar or 1D tensor."));
      const auto& sentence_in = t_sentence_in->vec<tstring>();
      const int32 b_size = t_sentence_in->dim_size(0);
      Tensor token_ids(DT_INT32, TensorShape({b_size, maxlen_}));
      Tensor paddings(DT_FLOAT, TensorShape({b_size, maxlen_}));
      auto t_token_ids = token_ids.tensor<int32, 2>();
      auto t_paddings = paddings.tensor<float, 2>();
      t_token_ids.setZero();
      t_paddings.setZero();
      int actual_maxlen = pad_to_maxlen_ ? maxlen_ : 0;
      for (int i = 0; i < b_size; ++i) {
        string sentence(sentence_in(i));
        VLOG(1) << "sentence " << sentence;
        std::vector<string> tokens;
        tokens =
            str_util::Split(sentence, delimiter_, str_util::SkipWhitespace());

        VLOG(1) << "#Tokens " << tokens.size() << " "
                << str_util::Join(tokens, "/");
        int cur_char = 0;
        for (const auto& token : tokens) {
          const int token_id = vocab_.TokenToId(token);
          // If the number of tokens is longer than the max length - truncate.
          if (cur_char >= maxlen_) {
            //            cur_char++;
            //            LOG(INFO) << "Sentence: \"" << sentence << "\"
            //            contained "
            //                      << tokens.size()
            //                      << " tokens, and was truncated to size: " <<
            //                      maxlen_
            //                      << " (" << tokens.size() - maxlen_
            //                      << " tokens were ignored).";
            break;
          }
          t_token_ids(i, cur_char) = token_id;
          t_paddings(i, cur_char) = 0.0;
          cur_char++;
        }
        actual_maxlen = std::max(actual_maxlen, cur_char);
        for (; cur_char < maxlen_; ++cur_char) {
          t_token_ids(i, cur_char) = pad_id_;
          t_paddings(i, cur_char) = 1.0;
        }
      }

      Tensor out_token_ids(DT_INT32, TensorShape({b_size, actual_maxlen}));
      Tensor out_paddings(DT_FLOAT, TensorShape({b_size, actual_maxlen}));

      typedef const Eigen::DSizes<Eigen::DenseIndex, 2> DSize2;
      out_token_ids.matrix<int32>() =
          t_token_ids.slice(DSize2{0, 0}, DSize2{b_size, actual_maxlen});
      out_paddings.matrix<float>() =
          t_paddings.slice(DSize2{0, 0}, DSize2{b_size, actual_maxlen});
      OP_REQUIRES_OK(ctx, ctx->set_output("token_ids", out_token_ids));
      OP_REQUIRES_OK(ctx, ctx->set_output("paddings", out_paddings));
    }
  }

 private:
  string vocab_filepath_;
  int maxlen_ = 0;
  bool pad_to_maxlen_ = true;
  string delimiter_ = " ";
  int pad_id_ = 0;
  Vocab vocab_;
};

REGISTER_KERNEL_BUILDER(Name("SentenceToIds").Device(DEVICE_CPU),
                        SentenceToIdsOp);
}  // namespace
}  // namespace delta
