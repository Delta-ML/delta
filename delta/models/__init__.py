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
"""Custom models."""

from delta.models.speech_cls_rawmodel import EmoCRNNRawModel
from delta.models.speech_cls_rawmodel import EmoDCRNNRawModel
from delta.models.speech_cls_rawmodel import EmoNDCRNNRawModel
from delta.models.speaker_cls_rawmodel import SpeakerCRNNRawModel
from delta.models.speech_cls_model import EmoCFNNModel
from delta.models.speech_cls_model import EmoCRNNModel
from delta.models.kws_model import TdnnKwsModel
from delta.models.asr_model import CTCAsrModel
from delta.models.text_seq_model import SeqclassCNNModel
from delta.models.text_seq_model import RnnAttentionModel
from delta.models.text_seq_model import TransformerModel
from delta.models.text_hierarchical_model import HierarchicalAttentionModel
from delta.models.text_match_model import MatchRnnTextClassModel
from delta.models.text_seq_label_model import BilstmCrfModel
