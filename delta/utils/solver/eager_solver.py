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

import delta.compat as tf
import numpy as np
import time
import os

from delta.data import dataset
from delta.data import dataloader
from delta import config
from delta.models import model_eager as model_lib
from delta import utils

FLAGS = config.FLAGS

layers = tf.keras.layers
tfe = tf.contrib.eager


def train_one_epoch(model,
                    optimizer,
                    dataset,
                    step_counter,
                    cmvn,
                    log_interval=None):
  assert len(cmvn) == 2
  start = time.time()
  for (batch, (feats, texts, labels, filenames,
               clip_ids)) in enumerate(dataset):
    feats = utils.apply_cmvn(feats, cmvn[0], cmvn[1])

    with tf.contrib.summary.record_summaries_every_n_global_steps(
        10, global_step=step_counter):
      with tf.GradientTape() as tape:
        logits = model(feats, training=True)
        loss_value = utils.losses(logits, labels, smoothing=0.0, is_train=True)
        tf.contrib.summary.scalar('loss', loss_value)
        tf.contrib.summary.scalar('accuracy', utils.accuracy(logits, labels))

      grads = tape.gradient(loss_value, model.variables)
      optimizer.apply_gradients(
          utils.clip_gradients(
              list(zip(grads, model.variables)),
              clip_ratio=FLAGS.clip_global_norm),
          global_step=step_counter)

      if log_interval and batch % log_interval == 0:
        rate = log_interval / (time.time() - start)
        print('Step #%d\tLoss: %0.6f (%d step/sec)' % (batch, loss_value, rate))
        start = time.time()


def eval(model, dataset, cmvn):
  avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
  accuracy = tfe.metrics.Accuracy('accuracy', dtype=tf.float32)

  for (batch, (feats, texts, labels, filenames,
               clip_ids)) in enumerate(dataset):
    feats = utils.apply_cmvn(feats, cmvn[0], cmvn[1])

    logits = model(feats, training=False)
    avg_loss(utils.losses(logits, labels, is_train=False))
    accuracy(
        tf.argmax(logits, axis=-1, output_type=tf.int64),
        tf.cast(labels, tf.int64))
    print("Eval set: Average loss: %0.4f, Accuracy: %4f%%\n" %
          (avg_loss.result(), 100 * accuracy.result()))

    with tf.contrib.summary.always_record_summaries():
      tf.contrib.summary.scalar('loss', avg_loss.result())
      tf.contrib.summary.scalar('accuracy', accuracy.result())


def print_vars(model):
  total = 0.0
  for var in model.variables:
    print(var.name, var.shape)
    if var.shape:
      total += np.prod(var.shape.as_list())
  print('\nTotal Parameters: %f' % (total))


def main():
  data = dataset.make_dataset('train', config.train_path,
                              config.train_textgrid_path, FLAGS)
  train_data = dataloader.input_func(
      data, FLAGS.batch_size, is_train=True, num_epoch=1)

  data = dataset.make_dataset('dev', config.dev_path, config.dev_textgrid_path,
                              FLAGS)
  dev_data = dataloader.input_func(data, FLAGS.batch_size, is_train=False)

  # create model and optimizer
  step_counter = tf.train.get_or_create_global_step()
  model = model_lib.Emotion(drop_rate=0.1)

  lr = tf.train.exponential_decay(
      FLAGS.learning_rate, step_counter, 100, FLAGS.decay_rate, staircase=True)
  optimizer = tf.train.AdamOptimizer(lr)
  print('init lr', lr().numpy())

  # checkpoint dirs
  if FLAGS.checkpoint:
    train_dir = os.path.join(FLAGS.checkpoint, 'train')
    eval_dir = os.path.join(FLAGS.checkpoint, 'eval')
    tf.gfile.MakeDirs(FLAGS.checkpoint)
  else:
    train_dir = None
    eval_dir = None
  summary_writer = tf.contrib.summary.create_file_writer(
      train_dir, flush_millis=10000)
  eval_summary_writer = tf.contrib.summary.create_file_writer(
      eval_dir, flush_millis=10000, name='eval')

  # create and restore checkpoint ( if one exists on the graph)
  checkpoint_prefix = os.path.join(FLAGS.checkpoint, 'ckpt')
  checkpoint = tf.train.Checkpoint(
      #  model=model, optimizer=optimizer, learning_rate=lr, step_counter=step_counter)
      model=model,
      optimizer=optimizer,
      step_counter=step_counter)
  # restore variables on creation if a checkpoint exists.
  stats = checkpoint.restore(tf.train.latest_checkpoint(FLAGS.checkpoint))
  #stats.assert_consumed()
  print('now lr', lr().numpy())

  cmvn = utils.load_cmvn(FLAGS.cmvn_path)

  device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'
  print("Using device %s" % (device))

  with tf.device(device):
    for e in range(FLAGS.num_epochs):
      # train
      start = time.time()
      with summary_writer.as_default():
        train_one_epoch(
            model, optimizer, train_data, step_counter, cmvn, log_interval=100)
      end = time.time()
      print(
          '\nTrain time for epoch #%d (%d total steps) (%f learning rate): %f' %
          (checkpoint.save_counter.numpy() + 1, step_counter.numpy(),
           lr().numpy(), end - start))

      if e == 0:
        print_vars(model)

      # eval
      with eval_summary_writer.as_default():
        eval(model, dev_data, cmvn)
      checkpoint.save(checkpoint_prefix)


if __name__ == '__main__':
  tf.enable_eager_execution()
  main()
