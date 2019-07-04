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

CUR_DIR := $(shell pwd)
EXC_DIR := ""

SUB_DIR := $(shell find . -maxdepth 1 -type d)
SUB_DIR := $(basename $(patsubst ./%,%,$(SUB_DIR)))
SUB_DIR := $(filter-out $(EXC_DIR),$(SUB_DIR))

MAKELIST = @for subdir in $(SUB_DIR); \
do \
$(MAKE) -C $(CUR_DIR)/$$subdir; \
done

CLEANLIST = @for subdir in $(SUB_DIR); \
do \
$(MAKE) -C $(CUR_DIR)/$$subdir clean; \
done

all:
	$(MAKELIST)

clean:
	$(CLEANLIST)
