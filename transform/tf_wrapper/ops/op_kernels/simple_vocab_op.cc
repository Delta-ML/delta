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

#include "../../../kernels/simple_vocab.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;  // NOLINT

namespace delta {

class VocabTokenToIdOp : public OpKernel {
 public:
  explicit VocabTokenToIdOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::vector<string> vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab", &vocab));
    bool load_token_ids_from_vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("load_token_ids_from_vocab",
                                     &load_token_ids_from_vocab));
    OP_REQUIRES_OK(ctx, vocab_.Load(vocab, load_token_ids_from_vocab));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* token;
    OP_REQUIRES_OK(ctx, ctx->input("token", &token));
    Tensor* id;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("id", token->shape(), &id));
    if (token->dims() == 0) {
      id->scalar<int32>()() = vocab_.TokenToId(token->scalar<tstring>()());
    } else {
      OP_REQUIRES(
          ctx, token->dims() == 1,
          errors::InvalidArgument("Input must be a scalar or 1D tensor."));
      for (int i = 0; i < token->dim_size(0); i++) {
        id->vec<int32>()(i) = vocab_.TokenToId(token->vec<tstring>()(i));
      }
    }
  }

 private:
  Vocab vocab_;
};

REGISTER_KERNEL_BUILDER(Name("VocabTokenToId").Device(DEVICE_CPU),
                        VocabTokenToIdOp);

class VocabIdToTokenOp : public OpKernel {
 public:
  explicit VocabIdToTokenOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::vector<string> vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab", &vocab));
    bool load_token_ids_from_vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("load_token_ids_from_vocab",
                                     &load_token_ids_from_vocab));
    OP_REQUIRES_OK(ctx, vocab_.Load(vocab, load_token_ids_from_vocab));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* id;
    OP_REQUIRES_OK(ctx, ctx->input("id", &id));
    Tensor* token;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("token", id->shape(), &token));
    if (id->dims() == 0) {
      token->scalar<tstring>()() = vocab_.IdToToken(id->scalar<int32>()());
    } else {
      OP_REQUIRES(
          ctx, id->dims() == 1,
          errors::InvalidArgument("Input must be a scalar or 1D tensor."));
      for (int i = 0; i < id->dim_size(0); i++) {
        token->vec<tstring>()(i) = vocab_.IdToToken(id->vec<int32>()(i));
      }
    }
  }

 private:
  Vocab vocab_;
};

REGISTER_KERNEL_BUILDER(Name("VocabIdToToken").Device(DEVICE_CPU),
                        VocabIdToTokenOp);

class TokenInVocabOp : public OpKernel {
 public:
  explicit TokenInVocabOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::vector<string> vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab", &vocab));
    bool load_token_ids_from_vocab;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("load_token_ids_from_vocab",
                                     &load_token_ids_from_vocab));
    OP_REQUIRES_OK(ctx, vocab_.Load(vocab, load_token_ids_from_vocab));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* token;
    OP_REQUIRES_OK(ctx, ctx->input("token", &token));
    Tensor* result;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("result", token->shape(), &result));
    if (token->dims() == 0) {
      result->scalar<bool>()() = vocab_.InVocab(token->scalar<tstring>()());
    } else {
      OP_REQUIRES(
          ctx, token->dims() == 1,
          errors::InvalidArgument("Input must be a scalar or 1D tensor."));
      for (int i = 0; i < token->dim_size(0); i++) {
        result->vec<bool>()(i) = vocab_.InVocab(token->vec<tstring>()(i));
      }
    }
  }

 private:
  Vocab vocab_;
};

REGISTER_KERNEL_BUILDER(Name("TokenInVocab").Device(DEVICE_CPU),
                        TokenInVocabOp);

}  // namespace delta
