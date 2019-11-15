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

#ifndef DELTANN_CORE_IO_H_
#define DELTANN_CORE_IO_H_

#include <memory>
#include <string>
#include <utility>

#include "core/buffer.h"
#include "core/misc.h"
#include "core/shape.h"

#ifdef USE_TF
#include "tensorflow/core/framework/tensor_shape.h"
#endif

using std::ostream;
using std::string;

namespace delta {

namespace core {

enum class InOut { DELTA_IN = 0, DELTA_OUT };

// model input/output metadata
class BaseInOut {
 public:
  BaseInOut(std::string name, int id, Shape shape, DataType dtype, InOut which)
      : _name(name), _id(id), _shape(shape), _dtype(dtype), _inout(which) {}

  virtual ~BaseInOut() {}

  int ndim(void) const { return _shape.ndim(); }

  std::string name() { return _name; }

  int id() { return _id; }

  Shape& shape(void) { return _shape; }

  const size_t size(void) const { return _shape.size(); }

  DataType dtype(void) const { return _dtype; }

  InOut inout_type(void) const { return _inout; }

  void set_dtype(const DataType dtype) { _dtype = dtype; }

#ifdef USE_TF
  void set_shape(const tensorflow::TensorShape& shape) {
    _shape.set_shape(shape);
  }
#endif
  void set_shape(const Shape& shape) { _shape.set_shape(shape); }

  friend std::ostream& operator<<(std::ostream& os, const BaseInOut& inout) {
    if (inout.inout_type() == InOut::DELTA_IN) {
      os << "Input: ";
    } else {
      os << "Output: ";
    }
    os << inout._name << " id:" << inout._id << " shape: " << inout._shape
       << " dtype: " << delta_dtype_str(inout._dtype);
    return os;
  }

 protected:
  InOut _inout;
  std::string _name;  // node name
  int _id;            // lite
  Shape _shape;
  DataType _dtype;
};

// graph input node
class Input : public BaseInOut {
 public:
  Input(std::string name, int id, Shape shape, DataType dtype)
      : BaseInOut(name, id, shape, dtype, InOut::DELTA_IN) {}
  Input(std::string name, int id, DataType dtype)
      : BaseInOut(name, id, Shape(), dtype, InOut::DELTA_IN) {}
  ~Input() {}
};

// graph output node
class Output : public BaseInOut {
 public:
  Output(std::string name, int id, Shape shape, DataType dtype)
      : BaseInOut(name, id, shape, dtype, InOut::DELTA_OUT) {}
  Output(std::string name, int id, DataType dtype)
      : BaseInOut(name, id, Shape(), dtype, InOut::DELTA_OUT) {}
  ~Output() {}
};

class BaseInOutData {
 public:
  explicit BaseInOutData(BaseInOut& inout) : _inout(inout) { allocate(); }

  void allocate(void) {
    if (this->dtype() != DataType::DELTA_NONE && 0 != this->size()) {
      _data = std::make_shared<Buffer>(this->bytes());
    } else {
      _data = std::make_shared<Buffer>();
    }
  }

  ~BaseInOutData() { _data = nullptr; }

  void* ptr() const { return _data->ptr(); }

  int ndim(void) const { return _inout.ndim(); }

  Shape& shape(void) { return _inout.shape(); }

  const Shape& shape(void) const { return _inout.shape(); }

  const size_t size(void) const { return _inout.size(); }

  DataType dtype() const { return _inout.dtype(); }

  const size_t bytes(void) const {
    return _inout.size() * delta_dtype_size(_inout.dtype());
  }

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
        make_random(static_cast<float*>(this->ptr()), this->size());
        break;
      case DataType::DELTA_INT32:
        std::fill_n(static_cast<int*>(this->ptr()), this->size(), 1);
        break;
      case DataType::DELTA_CHAR:
        for (int i = 0; i < size(); ++i) {
          char* ptr = static_cast<char*>(this->ptr());
          std::fill_n(ptr + i, 1, gen_random_char());
        }
        break;
      default:
        LOG_FATAL << "not support datatype";
    }
  }

  void resize(const std::size_t bytes) {
    DataType dtype = this->dtype();
    DELTA_CHECK(dtype != DataType::DELTA_NONE);
    _data->resize(bytes);
  }

  void copy_from(const void* src, std::size_t size) {
    DataType dtype = this->dtype();
    DELTA_CHECK(dtype != DataType::DELTA_NONE);
    std::size_t bytes = size * delta_dtype_size(dtype);
    this->resize(bytes);
    _data->copy_from(src, bytes);
  }

  void copy_from(const float* src) { copy_from(src, this->size()); }

  void copy_from(const float* src, std::size_t size) {
    DataType dtype = this->dtype();
    DELTA_CHECK(dtype != DataType::DELTA_NONE);
    std::size_t bytes = size * delta_dtype_size(dtype);
    this->resize(bytes);
    _data->copy_from(src, bytes);
  }

  void copy_to(void* dst, std::size_t byte_size) {
    _data->copy_to(dst, byte_size);
  }

#ifdef USE_TF
  tensorflow::TensorShape tensor_shape() const {
    tensorflow::Status status;
    const Shape& shape = this->shape();
    tensorflow::TensorShape ts;
    status =
        tensorflow::TensorShapeUtils::MakeShape(std::move(shape.vec()), &ts);
    if (!status.ok()) {
      LOG_FATAL << "Error when make shape from vector: " << shape.vec() << ", "
                << status;
    }
    return ts;
  }
#endif

  virtual BaseInOut& meta(void) const { return _inout; }

  friend std::ostream& operator<<(std::ostream& os, const BaseInOutData& data) {
    os << data.meta();
    os << "DataPtr: " << data.ptr();
    return os;
  }

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

#endif  // DELTANN_CORE_IO_H_
