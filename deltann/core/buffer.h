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

#ifndef DELTANN_CORE_BUFFER_H_
#define DELTANN_CORE_BUFFER_H_

#include <cstdlib>
#include "core/logging.h"
#include "core/misc.h"

namespace delta {

namespace core {

class Buffer {
 public:
  Buffer() : _size(0), _ptr(nullptr) {}

  explicit Buffer(const size_t size) : _size(size), _ptr(nullptr) {
    if (_size > 0) {
      _ptr = reinterpret_cast<void*>(std::malloc(_size));
    }
  }

  ~Buffer() {
    if (_ptr != nullptr) {
      std::free(_ptr);
      _ptr = nullptr;
      _size = 0;
    }
  }

  size_t resize(const size_t size) {
    if (!size) {
      LOG_INFO << "Warning, size is 0";
      return 0;
    }

    if (!_ptr) {
      _size = size;
      _ptr = reinterpret_cast<void*>(std::malloc(_size));
      return _size;
    }

    if (size > _size) {
      std::free(_ptr);
      _size = size;
      _ptr = reinterpret_cast<void*>(std::malloc(_size));
    }

    return _size;
  }

  void free() {
    if (_ptr != nullptr) {
      std::free(_ptr);
      _ptr = nullptr;
      _size = 0;
    }
  }

  void* ptr() { return _ptr; }

  size_t size() { return _size; }

  void zero() {
    if (nullptr != _ptr) {
      std::memset(_ptr, 0, _size);
    }
  }

  void copy_from(const void* src, const std::size_t size) {
    DELTA_CHECK(_ptr);
    DELTA_CHECK(src);
    DELTA_CHECK(size <= _size)
        << "expect size: " << size << " real size:" << _size;
    std::memcpy(_ptr, src, size);
  }

  void copy_to(void* dst, const std::size_t size) {
    DELTA_CHECK(_ptr);
    DELTA_CHECK(dst);
    DELTA_CHECK(size <= _size)
        << "expect size: " << size << " real size:" << _size;
    std::memcpy(dst, _ptr, size);
  }

 private:
  size_t _size;  // 当前的有效数据长度, 单位是类型char
  void* _ptr;

  DISALLOW_COPY_AND_ASSIGN(Buffer);
};

}  // namespace core

}  // namespace delta

#endif  // DELTANN_CORE_BUFFER_H_
