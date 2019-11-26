# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generate BUILD file for bazel build."""

import os

cppjieba = [
    'cppjieba/include/cppjieba/Jieba.hpp',
    'cppjieba/include/cppjieba/QuerySegment.hpp',
    'cppjieba/deps/limonp/Logging.hpp',
    'cppjieba/include/cppjieba/DictTrie.hpp',
    'cppjieba/deps/limonp/StringUtil.hpp',
    'cppjieba/deps/limonp/StdExtension.hpp',
    'cppjieba/include/cppjieba/Unicode.hpp',
    'cppjieba/deps/limonp/LocalVector.hpp',
    'cppjieba/include/cppjieba/Trie.hpp',
    'cppjieba/include/cppjieba/SegmentBase.hpp',
    'cppjieba/include/cppjieba/PreFilter.hpp',
    'cppjieba/include/cppjieba/FullSegment.hpp',
    'cppjieba/include/cppjieba/MixSegment.hpp',
    'cppjieba/include/cppjieba/MPSegment.hpp',
    'cppjieba/include/cppjieba/SegmentTagged.hpp',
    'cppjieba/include/cppjieba/PosTagger.hpp',
    'cppjieba/include/cppjieba/HMMSegment.hpp',
    'cppjieba/include/cppjieba/HMMModel.hpp',
    'cppjieba/include/cppjieba/KeywordExtractor.hpp',
]
copts = [
    "-Itensorflow/core/user_ops/ops",
    "-Itensorflow/core/user_ops/ops/cppjieba/include",
    "-Itensorflow/core/user_ops/ops/cppjieba/deps"
]

src = [
    os.path.join("kernels", one_path)
    for one_path in os.listdir("kernels")
    if one_path.endswith(".cc")
]
src += [
    os.path.join("kernels", one_path)
    for one_path in os.listdir("kernels")
    if one_path.endswith(".h")
]
src += cppjieba
# print(src)

first_line = 'load("//tensorflow:tensorflow.bzl",  "tf_custom_op_library")'
second_line = 'tf_custom_op_library(name = "x_ops.so", \nsrcs = ["{}"], \ncopts = ["{}"])'.format(
    '",\n"'.join(src), '",\n"'.join(copts))

print(first_line)
print(second_line)

with open("BUILD", "w") as f:
  f.write(first_line + "\n")
  f.write(second_line + "\n")

print("BUILD generated successfully!")
