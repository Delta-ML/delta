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

CXXFLAGS := -std=c++11 -fPIC -DINFERENCE_VERSION=\"$(shell git rev-parse --short HEAD)\" -DNDEBUG -O2

ABSEIL := $(ROOT_DIR)/../tools/abseil-cpp

INCLUDES := -I$(ROOT_DIR) -I$(JSONCPP) -I$(ABSEIL)

TEST_SRC := test.cc
TEST_BIN := test.bin

$(TEST_BIN): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TEST_SRC) -o $(TEST_BIN) -L $(LIBDIR) -ldeltann $(LIBS)

all: $(TEST_BIN)

clean:
	$(RM) $(TEST_BIN)
