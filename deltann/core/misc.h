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

#ifndef DELTANN_CORE_MISC_H_
#define DELTANN_CORE_MISC_H_

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/logging.h"

namespace delta {

static unsigned int RANDOM_SEED = 100;

enum class DeltaStatus { STATUS_OK = 0, STATUS_ERROR = 1 };

enum class DataType {
  DELTA_NONE = 0,
  DELTA_FLOAT16,
  DELTA_FLOAT32,
  DELTA_DOUBLE,
  DELTA_INT32,
  DELTA_CHAR,
  DELTA_STRING
};

enum class EngineType {
  DELTA_EIGINE_TF = 0,
  DELTA_EIGINE_TFTRT,
  DELTA_EIGINE_TFLITE,
  DELTA_EIGINE_TFSERVING,
  DELTA_EIGINE_SERVING,
};

/* The following code is from:
 * https://github.com/dmlc/dmlc-core/blob/master/include/dmlc/base.h
 */

/*! \brief whether or not use c++11 support */
#ifndef DELTA_USE_CXX11
#if defined(__GXX_EXPERIMENTAL_CXX0X__) || defined(_MSC_VER)
#define DELTA_USE_CXX11 1
#else
#define DELTA_USE_CXX11 (__cplusplus >= 201103L)
#endif
#endif

std::size_t delta_dtype_size(DataType type);

DataType delta_str_dtype(const std::string& type);

std::string delta_dtype_str(DataType type);

extern std::unordered_map<std::string, EngineType> _global_engines;

void make_random(float* ptr, int len);

char gen_random_char();

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  out << "{";
  size_t last = v.size() - 1;
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    if (i != last) {
      out << ", ";
    }
  }
  out << "}";
  return out;
}

}  // namespace delta
#endif  // DELTANN_CORE_MISC_H_
