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
#include <cstddef>
#include <random>

#include "core/utils/logging.h"
#include "core/utils/misc.h"

namespace delta {

std::size_t delta_sizeof(DataType type) {
  switch (type) {
    case DataType::DELTA_FLOAT32:
      return sizeof(float);
    case DataType::DELTA_FLOAT16:
      return sizeof(short);
    case DataType::DELTA_DOUBLE:
      return sizeof(double);
    case DataType::DELTA_INT32:
      return sizeof(std::int32_t);
    case DataType::DELTA_CHAR:
      return sizeof(char);
    default:
      assert(false);
      return 0;
  }
}

DataType delta_type_switch(const std::string& type) {
  if (type == "float") {
    return DataType::DELTA_FLOAT32;
  } else if (type == "double") {
    return DataType::DELTA_DOUBLE;
  } else if (type == "int") {
    return DataType::DELTA_INT32;
  } else if (type == "char") {
    return DataType::DELTA_CHAR;
  } else {
    LOG_FATAL << "Unknown data type " << type;
    return DataType::DELTA_NONE;
  }
}

std::unordered_map<std::string, EngineType> _global_engines{
    {"TF", EngineType::DELTA_EIGINE_TF},
    {"TFTRT", EngineType::DELTA_EIGINE_TFTRT},
    {"TFLITE", EngineType::DELTA_EIGINE_TFLITE},
    {"TFSERVING", EngineType::DELTA_EIGINE_TFSERVING},
    {"SERVING", EngineType::DELTA_EIGINE_SERVING}};

void make_random(float* ptr, int len) {
  static std::mt19937 random_engine;
  std::uniform_real_distribution<float> distribution(-1, 1);
  for (int i = 0; i < len; ++i) {
    ptr[i] = distribution(random_engine);
  }
}

char gen_random_char() {
  // return (char)(random(127-33) + 33);
  char c = 'A' + rand_r(&RANDOM_SEED) % 24;
  return c;
}


}  // namespace delta
