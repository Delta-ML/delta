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

#include <string>

#include "core/shape.h"
#include "core/misc.h"

#ifdef USE_TF
#include "tensorflow/core/framework/tensor_shape.h"
#endif

using std::string;

namespace delta {

namespace core {

enum class InOut { DELTA_IN = 0, DELTA_OUT };

// model input/output identifier
class BaseInOut {
 public:
  BaseInOut(std::string name, int id, Shape shape, DataType dtype)
      : _name(name), _id(id), _shape(shape), _dtype(dtype) {}

  virtual ~BaseInOut() {}

  int ndim(void) const { return _shape.ndim(); }

  std::string name() { return _name; }

  int id() { return _id; }

  const Shape& shape(void) const { return _shape; }

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
      : BaseInOut(name, id, shape, dtype) {
    _inout = InOut::DELTA_IN;
  }

  Input(std::string name, int id, DataType dtype)
      : BaseInOut(name, id, Shape(), dtype) {
    _inout = InOut::DELTA_IN;
  }

  ~Input() {}
};

// graph output node
class Output : public BaseInOut {
 public:
  Output(std::string name, int id, Shape shape, DataType dtype)
      : BaseInOut(name, id, shape, dtype) {
    _inout = InOut::DELTA_OUT;
  }

  Output(std::string name, int id, DataType dtype)
      : BaseInOut(name, id, Shape(), dtype) {
    _inout = InOut::DELTA_OUT;
  }

  ~Output() {}
};

}  // namespace core

}  // namespace delta

#endif  // DELTANN_CORE_IO_H_
