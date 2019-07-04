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

#include "dynamic_loader.h"
#include <util/logging.h>

namespace inference {

static std::string default_lib_path = "lib";

static inline std::string join(const std::string& part1,
                               const std::string& part2) {
  /// directory separator
  const char sep = '/';
  if (!part2.empty() && part2.front() == sep) {
    return part2;
  }
  std::string ret;
  ret.reserve(part1.size() + part2.size() + 1);
  ret = part1;
  if (!ret.empty() && ret.back() != sep) {
    ret += sep;
  }
  ret += part2;

  return ret;
}

static inline void get_dsohandle(const std::string& dso_name,
                                 void** dso_handle) {
  int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
  *dso_handle = nullptr;

  // open
  *dso_handle = dlopen(dso_name.c_str(), dynload_flags);

  if (nullptr == *dso_handle) {
    BFATAL << "Failed to open dynamic library: " << dso_name;
  } else {
    BLOG << "Success to open dynamic library: " << dso_name;
  }
}

void get_feature_extraction_dsohandle(void** dso_handle) {
  std::string feature_so_name;
  if (getenv("LIB_FEATURE_SO_PATH")) {
    std::string path = getenv("LIB_FEATURE_SO_PATH");
    feature_so_name = join(path, "libfeature_extraction.so");
  } else {
    feature_so_name = join(default_lib_path, "libfeature_extraction.so");
  }

  if (feature_so_name.empty()) {
    BFATAL << "Failed to find dynamic feature extraction library: "
              "libfeature_extraction.so";
  }

  get_dsohandle(feature_so_name, dso_handle);
}

}  // end namespace inference
