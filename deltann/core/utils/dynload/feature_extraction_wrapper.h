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

#ifndef DYNLOAD_FEATURE_EXTRACTION_WRAPPER_H
#define DYNLOAD_FEATURE_EXTRACTION_WRAPPER_H

#include "dynload/dynamic_loader.h"
#include "filterbank.h"

namespace inference {

namespace dynload {

using namespace FeatureExtraction;

extern std::once_flag feature_dso_flag;
extern void* feature_dso_handle;

#define DYNAMIC_LOAD_FEATURE_WRAP(__name)                                \
  struct dynload__##__name {                                             \
    template <typename... Args>                                          \
    auto operator()(Args... args) -> decltype(__name(args...)) {         \
      using feature_func = decltype(__name(args...)) (*)(Args...);       \
      std::call_once(feature_dso_flag, get_feature_extraction_dsohandle, \
                     &feature_dso_handle);                               \
      static void* p_##__name = dlsym(feature_dso_handle, #__name);      \
      return reinterpret_cast<feature_func>(p_##__name)(args...);        \
    }                                                                    \
  };                                                                     \
  extern struct dynload__##__name __name

#define FEATURE_EXTRACTION_ROUTINE_EACH(__macro) \
  __macro(fbank_create);                         \
  __macro(fbank_extract_feat);                   \
  __macro(fbank_destroy);

FEATURE_EXTRACTION_ROUTINE_EACH(DYNAMIC_LOAD_FEATURE_WRAP)

#undef DYNAMIC_LOAD_FEATURE_WRAP

}  // namespace dynload

#define FBANK_CREATE dynload::fbank_create
#define FBANK_EXTRACTFEAT dynload::fbank_extract_feat
#define FBANK_DESTROY dynload::fbank_destroy

}  // namespace inference

#endif
