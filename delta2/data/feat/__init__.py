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
''' speech feature '''
from delta.data.feat import speech_ops

from .speech_feature import load_wav
from .speech_feature import extract_feature
from .speech_feature import add_delta_delta

# numpy
from .speech_feature import extract_fbank
from .speech_feature import delta_delta
from .speech_feature import fbank_feat
from .speech_feature import powspec_feat
from .speech_feature import extract_feat
