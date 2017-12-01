from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import csv

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops





class SubmissionProcessor(object):

  def __init__(self, data_dir):
    self.data_dir = data_dir
    self.prepare_data_index()


  def prepare_data_index(self):
    self.data_index = []
    with open('./sample_submission.csv') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            self.data_index.append('/home/vitaly/competition/test/audio/' + row[0])
        self.data_index = self.data_index[1:]
  def write_to_csv(self, human_string):
      with open('./baseline_submission.csv', 'w') as csvfile:
          spamwriter = csv.writer(csvfile)
          spamwriter.writerow(['fname'] + ['label'])
          for i in range(len(human_string)):
            spamwriter.writerow([self.data_index[i].rsplit('/', 1)[1]] + [human_string[i]])

  def get_test_data(self, how_many, offset, model_settings):

    candidates = self.data_index
    if how_many == -1:
      sample_count = len(candidates)
    else:
        sample_count = max(0, min(how_many, len(candidates) - offset))
    desired_samples = model_settings['desired_samples']
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
      spectrogram = contrib_audio.audio_spectrogram(
          wav_decoder.audio,
          window_size=model_settings['window_size_samples'],
          stride=model_settings['window_stride_samples'],
          magnitude_squared=True)
      mfcc = contrib_audio.mfcc(
          spectrogram,
          wav_decoder.sample_rate,
          dct_coefficient_count=model_settings['dct_coefficient_count'])
      for i in range(offset, offset + sample_count):
        input_dict = {wav_filename_placeholder : candidates[i]}
        data[i - offset, :] = sess.run(mfcc, feed_dict=input_dict).flatten()
    return data
