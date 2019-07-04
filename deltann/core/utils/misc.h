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

#ifndef DELTANN_UTILS_MISC_H_
#define DELTANN_UTILS_MISC_H_

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace delta {

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

#define DELTA_CHECK(x) \
  if (!(x)) LOG_FATAL << "Check failed: " #x << ' '

#define DELTA_CHECK_LT(x, y) DELTA_CHECK((x) < (y))
#define DELTA_CHECK_GT(x, y) DELTA_CHECK((x) > (y))
#define DELTA_CHECK_LE(x, y) DELTA_CHECK((x) <= (y))
#define DELTA_CHECK_GE(x, y) DELTA_CHECK((x) >= (y))
#define DELTA_CHECK_EQ(x, y) DELTA_CHECK((x) == (y))
#define DELTA_CHECK_NE(x, y) DELTA_CHECK((x) != (y))

#define DELTA_ASSERT_OK(status) DELTA_CHECK_EQ(status, DeltaStatus::STATUS_OK)

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

/*!
 * \brief Disable copy constructor and assignment operator.
 *
 * If C++11 is supported, both copy and move constructors and
 * assignment operators are deleted explicitly. Otherwise, they are
 * only declared but not implemented. Place this macro in private
 * section if C++11 is not available.
 */
#ifndef DISALLOW_COPY_AND_ASSIGN
#if DELTA_USE_CXX11
#define DISALLOW_COPY_AND_ASSIGN(T) \
  T(T const&) = delete;             \
  T(T&&) = delete;                  \
  T& operator=(T const&) = delete;  \
  T& operator=(T&&) = delete
#else
#define DISALLOW_COPY_AND_ASSIGN(T) \
  T(T const&);                      \
  T& operator=(T const&)
#endif
#endif

std::size_t delta_sizeof(DataType type);

DataType delta_type_switch(const std::string& type);

extern std::unordered_map<std::string, EngineType> _global_engines;

void make_random(float* ptr, int len);

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
#endif  // DELTANN_UTILS_MISC_H_
