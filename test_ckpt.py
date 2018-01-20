# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
r"""Simple speech recognition to spot a limited number of keywords.

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import os

import numpy as np
import pandas as pd
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import json

from input_data import *
from models import *
from tensorflow.python.platform import gfile

from input_data import *
import submission_processor
from models import *

FLAGS = None

def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]

def write_outputs_to_file(predictions, final_fc, probs, wav_files, model_settings, truth=None):
  if truth is not None:
    if not os.path.exists('correctly_predicted_out'):
      os.makedirs('correctly_predicted_out')
    if not os.path.exists('errors_out'):
      os.makedirs('errors_out')
  else:
    if not os.path.exists('features'):
      os.makedirs('features')
  path_to_labels = model_settings['path_to_labels']
  labels = load_labels(path_to_labels)
  for id_x, value in enumerate(predictions):
    if truth is None:
      target_folder = 'features'
    else:
      if (predictions[id_x] == truth[id_x]) or ((predictions[id_x] == 1) and (labels[truth[id_x]] not in labels)):
        target_folder = 'correctly_predicted_out'
      else:
        target_folder = 'errors_out'
    with open('%s/%s.features' % (target_folder, os.path.basename(wav_files[id_x])), 'w') as feature_file:
      feature_file.write('audio_sample_name ' + wav_files[id_x] + '\n')
      feature_file.write('fc ' + " ".join(str(x) for x in final_fc[id_x]) + '\n')
      feature_file.write('probs ' + " ".join(str(x) for x in probs[id_x]) + '\n')
      feature_file.write('predicted_label ' + str(labels[predictions[id_x]]) + '\n')
      if truth is not None:
        feature_file.write('true_label ' + labels[truth[id_x]] + '\n')
      feature_file.write('labels : ' + " ".join(x for x in labels))
    feature_file.close()

def main(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()

  model_settings = prepare_model_settings(FLAGS.arch_config_file)
  audio_processor = AudioProcessor(FLAGS.data_url, FLAGS.data_dir, model_settings)
  model_settings['noise_label_count'] = audio_processor.background_label_count() + 1

  graph = Graph(model_settings)
  tf.summary.scalar('accuracy', graph.evaluation_step)

  global_step = tf.contrib.framework.get_or_create_global_step()
  tf.global_variables_initializer().run()

  if FLAGS.start_checkpoint:
    graph.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = global_step.eval(session=sess)

  # Save list of words.
  with gfile.GFile(
      os.path.join(FLAGS.checkpoint_dir, graph.get_arch_name() + '_labels.txt'),
      'w') as f:
    f.write('\n'.join(audio_processor.words_list))

  batch_size = int(model_settings['batch_size'])

  # write network outputs for test data into file
  set_size = audio_processor.set_size('validation')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_conf_matrix = None
  for i in xrange(0, set_size, batch_size):
    test_fingerprints, test_ground_truth, noise_labels, wav_files = audio_processor.get_data(
        batch_size, i, model_settings, 0.0, 0.0, 0, 'validation', sess, features=model_settings['features'])

    test_accuracy, conf_matrix, final_fc, probs, predictions, truth = sess.run(
        [graph.evaluation_step, graph.confusion_matrix,
         graph.final_fc, graph.probabilities, graph.predicted_indices, graph.expected_indices],
        feed_dict={
            graph.fingerprint_input: test_fingerprints,
            graph.ground_truth_input: test_ground_truth,
            graph.is_training: 0,
            graph.dropout_prob: 1.0
        })
    wav_paths = []
    for file_dict in wav_files:
      wav_paths.append(file_dict['file'])
    write_outputs_to_file(predictions, final_fc, probs, wav_paths, model_settings, truth)

    bs = min(batch_size, set_size - i)
    total_accuracy += (test_accuracy * bs) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100, set_size))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        default='http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/speech_dataset/',
        help="""\
        Where to download the speech training data to.
        """)
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='/tmp/speech_commands_train',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--arch_config_file',
        type=str,
        default='model_configs/dummy_conf',
        help='File containing model parameters')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
