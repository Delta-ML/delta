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

#include "cppjieba/Jieba.hpp"  // NOLINT

#include <string>  // NOLINT
#include <vector>  // NOLINT

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

using namespace tensorflow;  // NOLINT
using std::string;
using std::vector;

namespace delta {

class JiebaCutOp : public OpKernel {
 public:
  explicit JiebaCutOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    bool use_file;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_file", &use_file));
    if (use_file) {
      std::string dict_path;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("dict_path", &dict_path));
      std::string hmm_path;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("hmm_path", &hmm_path));
      std::string user_dict_path;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("user_dict_path", &user_dict_path));
      std::string idf_path;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("idf_path", &idf_path));
      std::string stop_word_path;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("stop_word_path", &stop_word_path));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("hmm", &hmm));
      jieba_ = new cppjieba::Jieba(dict_path, hmm_path, user_dict_path,
                                   idf_path, stop_word_path);
    } else {
      std::vector<std::string> dict_lines;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("dict_lines", &dict_lines));
      std::vector<std::string> model_lines;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("model_lines", &model_lines));
      std::vector<std::string> user_dict_lines;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("user_dict_lines", &user_dict_lines));
      std::vector<std::string> idf_lines;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("idf_lines", &idf_lines));
      std::vector<std::string> stop_word_lines;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("stop_word_lines", &stop_word_lines));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("hmm", &hmm));
      jieba_ = new cppjieba::Jieba(dict_lines, model_lines, user_dict_lines,
                                   idf_lines, stop_word_lines);
    }
  }

  ~JiebaCutOp() { delete jieba_; }

  void Compute(OpKernelContext* ctx) override {
    std::vector<string> words;
    const Tensor* t_sentence_in;
    OP_REQUIRES_OK(ctx, ctx->input("sentence_in", &t_sentence_in));
    Tensor* t_sentence_out;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("sentence_out", t_sentence_in->shape(),
                                        &t_sentence_out));
    if (t_sentence_in->dims() == 0) {
      jieba_->Cut(t_sentence_in->scalar<tstring>()(), words, hmm);
      t_sentence_out->scalar<tstring>()() = str_util::Join(words, " ");
    } else {
      OP_REQUIRES(
          ctx, t_sentence_in->dims() == 1,
          errors::InvalidArgument("Input must be a scalar or 1D tensor."));
      int batch = t_sentence_in->dim_size(0);
      for (int i = 0; i < batch; i++) {
        jieba_->Cut(t_sentence_in->vec<tstring>()(i), words, hmm);
        t_sentence_out->vec<tstring>()(i) = str_util::Join(words, " ");
      }
    }
  }

 private:
  cppjieba::Jieba* jieba_;
  bool hmm;
};

REGISTER_KERNEL_BUILDER(Name("JiebaCut").Device(DEVICE_CPU), JiebaCutOp);
}  // namespace delta
