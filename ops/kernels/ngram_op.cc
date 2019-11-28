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

#include <assert.h>

#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define REGISTER_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Ngram").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      NgramOp<type>)

namespace delta {

using namespace tensorflow;  // NOLINT
using std::vector;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

const int PAD_IDX = 0;

template <typename T>
// self-defined op, should override Compute() function.
class NgramOp : public OpKernel {
 public:
  explicit NgramOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("word_ngrams", &word_ngrams));
    OP_REQUIRES_OK(context, context->GetAttr("vocab_size", &vocab_size));
    OP_REQUIRES_OK(context, context->GetAttr("bucket_size", &bucket_size));
  }

  // hash function, from fasttext source code
  void NgramHash(vector<int>* line, const vector<int>& hashes, int n,
                 int nwords_, int bucket) const {
    for (int i = 0; i < hashes.size(); i++) {
      uint64_t h = hashes[i];
      for (int j = i + 1; j < hashes.size() && j < i + n; j++) {
        h = h * 116049371 + hashes[j];
        line->push_back(nwords_ + h % bucket);
      }
    }
  }

  int GetOutputLen(int input_len, int word_ngrams) {
    if (input_len <= 0) {
      return 0;
    } else if (word_ngrams <= 1) {
      return input_len;
    }
    int x = input_len > word_ngrams ? input_len - word_ngrams + 1 : 1;
    int l = input_len - x + 1;
    return ((input_len + x) * l) >> 1;
  }

  void Compute(OpKernelContext* context) override {
    // get input tensor from context
    const Tensor& input_tensor = context->input(0);

    const int ndims = input_tensor.dims();

    if (ndims < 1) {
      auto input_flat = input_tensor.flat<T>();
      int input_len = input_tensor.dim_size(0);
      int output_len = GetOutputLen(input_len, word_ngrams);

      TensorShape out_shape({output_len});

      // create the ouput_tensor, use context->allocate_ouput() to allocate the
      // memory.
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, out_shape, &output_tensor));
      auto output_flat = output_tensor->flat<T>();

      vector<int> input_vec, ngram_vec;
      for (int i = 0; i < input_len; ++i) {
        input_vec.push_back(input_flat(i));
      }
      NgramHash(&ngram_vec, input_vec, word_ngrams, vocab_size, bucket_size);

      for (int i = 0; i < output_len; i++) {
        if (i < input_len) {
          output_flat(i) = input_vec[i];
        } else {
          output_flat(i) = ngram_vec[i - input_len];
        }
      }
    } else {
      TensorShape out_shape;
      for (int i = 0; i < ndims - 1; ++i) {
        out_shape.AddDim(input_tensor.dim_size(i));
      }

      auto n_batch = (ndims == 1) ? 1 : out_shape.num_elements();
      int sent_len = static_cast<int>(input_tensor.dim_size(ndims - 1));
      int ngram_len = static_cast<int>(GetOutputLen(sent_len, word_ngrams));
      //            printf("n_batch: %d, sent_len: %d, ngram_len: %d \n",
      //            (int)n_batch, sent_len, ngram_len);

      Tensor input_reshaped;
      CHECK(input_reshaped.CopyFrom(input_tensor,
                                    TensorShape({n_batch, sent_len})));

      out_shape.AddDim(ngram_len);

      // create the ouput_tensor, use context->allocate_ouput() to allocate the
      // memory.
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, out_shape, &output_tensor));

      Tensor output_reshaped;
      CHECK(output_reshaped.CopyFrom(*output_tensor,
                                     TensorShape({n_batch, ngram_len})));

      auto input_flat = input_reshaped.flat<T>();
      auto output_flat = output_reshaped.flat<T>();

      for (int batch = 0; batch < n_batch; ++batch) {
        vector<int> input_vec, ngram_vec;
        int offset = batch * sent_len;
        int end_idx = offset + sent_len - 1;  // last index of the sentence
        while (end_idx > offset) {
          if (static_cast<int>(input_flat(end_idx)) != PAD_IDX) break;
          end_idx--;
        }
        for (int i = offset; i <= end_idx; ++i) {
          input_vec.push_back(static_cast<int>(input_flat(i)));
        }
        NgramHash(&ngram_vec, input_vec, word_ngrams, vocab_size, bucket_size);

        offset = batch * ngram_len;
        for (int i = 0; i < ngram_len; i++) {
          int idx = offset + i;
          int real_ngram_len = input_vec.size() + ngram_vec.size();
          if (i < input_vec.size()) {
            output_flat(idx) = input_vec[i];
          } else if (i < real_ngram_len) {
            output_flat(idx) = ngram_vec[i - input_vec.size()];
          } else {
            output_flat(idx) = PAD_IDX;
          }
        }
      }
    }
  }

 private:
  int vocab_size;
  int bucket_size;
  int word_ngrams;
};  // NgramOp

REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);

}  // namespace delta
