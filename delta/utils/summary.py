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
"""Summary related utilities."""

import delta.compat as tf


def flush(writer=None, name=None):
  """Flush"""
  if tf.executing_eagerly():
    tf.contrib.summary.flush(writer, name)
  else:
    tf.summary.flush(writer, name)  # pylint: disable=no-member


def scalar(name, value):  # pylint: redefined-outer-name
  "Scalar"
  if tf.executing_eagerly():
    tf.contrib.summary.scalar(name, value)
  else:
    tf.summary.scalar(name, value)


def histogram(name, values):
  "Histogram"
  if tf.executing_eagerly():
    tf.contrib.summary.histogram(name, values)
  else:
    tf.summary.histogram(name, values)


def text(name, tensor):
  "Text"
  if tf.executing_eagerly():
    tf.contrib.summary.text(name, tensor)
  else:
    tf.summary.text(name, tensor)


def audio(name, tensor, sample_rate, max_outputs=3):
  "Audio"
  if tf.executing_eagerly():
    tf.contrib.summary.audio(name, tensor, sample_rate, max_outputs)
  else:
    tf.summary.audio(name, tensor, sample_rate, max_outputs)


def image(name, tensor, max_images=3):
  "Image"
  if tf.executing_eagerly():
    tf.contrib.summary.image(name, tensor, max_images=max_images)
  else:
    tf.summary.image(name, tensor, max_outputs=max_images)


# pylint: disable=too-many-arguments
def summary_writer(logdir,
                   graph=None,
                   max_queue=10,
                   flush_secs=120,
                   graph_def=None,
                   filename_suffix=None,
                   session=None,
                   name=None):
  """Summary writer."""
  if tf.executing_eagerly():
    return tf.contrib.summary.create_file_writer(
        logdir,
        max_queue=max_queue,
        flush_millis=flush_secs * 1000,
        filename_suffix=filename_suffix,
        name=name)
  return tf.summary.FileWriter(logdir, graph, max_queue, flush_secs, graph_def,
                               filename_suffix, session)
