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

#ifndef DELTANN_CORE_DATA_H_
#define DELTANN_CORE_DATA_H_

#include <memory>
#include <string>
#include <vector>

#include "core/buffer.h"
#include "core/io.h"
#include "core/misc.h"
#include "core/shape.h"

#ifdef USE_TF
#include "tensorflow/core/framework/tensor_shape.h"
#endif

namespace delta {

namespace core {

class BaseInOutData {
 public:
  explicit BaseInOutData(BaseInOut& inout) : _inout(inout) {
    if (inout.dtype() != DataType::DELTA_NONE && 0 != inout.size()) {
      _data = std::make_shared<Buffer>(inout.size() *
                                       delta_dtype_size(inout.dtype()));
    } else {
      _data = std::make_shared<Buffer>();
    }
  }

  ~BaseInOutData() { _data = nullptr; }

  void* ptr() const { return _data->ptr(); }

  int ndim(void) const { return _inout.ndim(); }

  const Shape& shape(void) const { return _inout.shape(); }

  const size_t size(void) const { return _inout.size(); }

  const size_t byte_size(void) const {
    return _inout.size() * delta_dtype_size(_inout.dtype());
  }

  DataType dtype() const { return _inout.dtype(); }

  const std::string name() const { return _inout.name(); }

  int id() const { return _inout.id(); }

  void set_shape(const Shape& shape) { _inout.set_shape(shape); }

#ifdef USE_TF
  void set_shape(const tensorflow::TensorShape& shape) {
    _inout.set_shape(shape);
  }
#endif

  void set_dtype(const DataType dtype) { _inout.set_dtype(dtype); }

  void feed_random_data() {
    DataType dtype = this->dtype();
    switch (dtype) {
      case DataType::DELTA_FLOAT32:
        make_random(static_cast<float*>(_data->ptr()), size());
        break;
      case DataType::DELTA_INT32:
        std::fill_n(static_cast<int*>(_data->ptr()), size(), 1);
        break;
      case DataType::DELTA_CHAR:
        for (int i = 0; i < size(); ++i) {
          char* ptr = static_cast<char*>(_data->ptr());
          std::fill_n(ptr + i, 1, gen_random_char());
        }
        break;
      default:
        LOG_FATAL << "not support datatype";
    }
  }

  void copy_from(const void* src, std::size_t size) {
    DataType dtype = this->dtype();
    if (dtype != DataType::DELTA_NONE) {
      std::size_t bytes = size * delta_dtype_size(dtype);
      resize(bytes);
      _data->copy_from(src, bytes);
    } else {
      LOG_FATAL << "_dtype is DataType::DELTA_NONE ";
    }
  }

  void copy_from(const float* src) { copy_from(src, this->size()); }

  void copy_from(const float* src, std::size_t size) {
    DataType dtype = this->dtype();
    if (dtype != DataType::DELTA_NONE) {
      _data->copy_from(src, size * delta_dtype_size(dtype));
    } else {
      LOG_FATAL << "_dtype is DataType::DELTA_NONE ";
    }
  }

  void copy_to(void* dst, std::size_t byte_size) {
    _data->copy_to(dst, byte_size);
  }

  void resize(const std::size_t size) {
    DataType dtype = this->dtype();
    if (dtype != DataType::DELTA_NONE) {
      _data->resize(size * delta_dtype_size(dtype));
    } else {
      LOG_FATAL << "Data type is DELTA_NONE";
    }
  }

#ifdef USE_TF
  tensorflow::TensorShape tensor_shape() const {
    tensorflow::Status status;
    const Shape& shape = this->shape();
    int ndim = this->ndim();
    std::vector<int> tmp_shape;
    for (int i = 0; i < ndim; ++i) {
      tmp_shape.push_back(shape[i]);
    }
    tensorflow::TensorShape ts;
    status = tensorflow::TensorShapeUtils::MakeShape(tmp_shape, &ts);
    if (!status.ok()) {
      LOG_FATAL << "Error when make shape from vector: " << tmp_shape << ", "
                << status;
    }
    return ts;
  }
#endif

 protected:
  BaseInOut& _inout;
  std::shared_ptr<Buffer> _data;
};

class InputData : public BaseInOutData {
 public:
  explicit InputData(BaseInOut& inout) : BaseInOutData(inout) {}
};

class OutputData : public BaseInOutData {
 public:
  explicit OutputData(BaseInOut& inout) : BaseInOutData(inout) {}
};

}  // namespace core

}  // namespace delta

#endif  // DELTANN_CORE_DATA_H_
