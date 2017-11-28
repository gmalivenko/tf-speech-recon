import random
import os
import numpy as np
import array
import struct

from python_speech_features import mfcc
import webrtcvad
import scipy.io.wavfile as wav
# import matplotlib.pyplot as plt

from wav_crop import crop_wav

import h5py

# from pydub import AudioSegment,silence

#empiricaly calcullated from test set
MINIMUM_FEATURE_SIZE = 100


class FeatureExtractor:
    def __init__(self):
        self.sample_names = []
        self.dir_names = []

    def extract_features(self, sound_path, feature_path):
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)

        h5f_train = h5py.File(feature_path + 'train_features.h5', 'w')
        h5f_test = h5py.File(feature_path + 'test_features.h5', 'w')

        for (dir_path, dir_names, filen_ames) in os.walk(sound_path):
            self.dir_names.extend(dir_names)
            break

        for dir_name in self.dir_names:
            print('Processing ' + sound_path + dir_name)
            wav_files = []

            for (dir_path, dir_names, file_names) in os.walk(sound_path + dir_name):
                wav_files.extend(file_names)
                break

            for id, file_name in enumerate(wav_files):
                sample_full_name = sound_path + dir_name + '/' + file_name
                (rate, sig) = wav.read(sample_full_name)

                #cut_signal
                signal_cut = []
                raw_cut = crop_wav(sig, rate)
                for i, s in enumerate(raw_cut):
                    signal_cut.extend(array.array("h", s))

                if len(signal_cut) == 0:
                    signal_cut = sig

                signal_cut = np.array(signal_cut)
                mfcc_feat = mfcc(signal_cut, rate)
                mfcc_padded = self.padding(mfcc_feat)

                if id < 0.9 * len(wav_files):
                    h5f_train.create_dataset(dir_name + '/' + str(id), data=mfcc_padded)
                else:
                    h5f_test.create_dataset(dir_name + '/' + str(id), data=mfcc_padded)
                # np.savetxt(feature_path + dir_name + '/' + file_name + '.feature', mfcc_feat, delimiter=',')
                # print(str((100 * id)/len(self.sample_names)) + '%')

        h5f_test.close()
        h5f_train.close()

    def sample(self, size):
        sample_id = random.sample(range(len(self.sample_names)), min(size, len(self.sample_names)))
        for i in range(len(sample_id)):
            sample_file_name = self.path + self.sample_names[sample_id[i]]
            (rate, sig) = wav.read(sample_file_name)

            signal_cut = []
            # mfcc_feat = mfcc(sig, rate)
            raw_cut = crop_wav(sig, rate)

            for i, s in enumerate(raw_cut):
                print(s)
                print(array.array("h", s))
                signal_cut.extend(array.array("h", s))

            if len(signal_cut) == 0:
                signal_cut = sig

            signal_cut = np.array(signal_cut)

            mfcc_feat = mfcc(signal_cut, rate, numcep=13)

        return mfcc_feat


    def padding(self, mfcc_feat):
        for i in range(MINIMUM_FEATURE_SIZE - np.shape(mfcc_feat)[0]):
            mfcc_feat = np.concatenate((mfcc_feat, np.zeros((1, np.shape(mfcc_feat)[1]))), axis=0)
        return mfcc_feat
    #
    # def visualize(self, path):
    #     for (dirpath, dirnames, filenames) in os.walk(path):
    #         self.sample_names.extend(filenames)
    #         break
    #
    #     id = random.sample(range(len(self.sample_names)), 1)[0]
    #     sample_file_name = path + self.sample_names[id]
    #     # sample_file_name = self.path + "0f7dc557_nohash_2.wav"
    #     (rate, sig) = wav.read(sample_file_name)
    #
    #     print(sample_file_name)
    #
    #     fig = plt.figure(figsize=(14, 8))
    #     ax1 = fig.add_subplot(211)
    #     ax1.set_title('Raw wave of ' + sample_file_name)
    #     ax1.set_ylabel('Amplitude')
    #     ax1.plot(sig)
    #
    #     signal_cut = []
    #     # mfcc_feat = mfcc(sig, rate)
    #     raw_cut = crop_wav(sig, rate)
    #
    #     for i,s in enumerate(raw_cut):
    #         print(s)
    #         print(array.array("h", s))
    #         signal_cut.extend(array.array("h", s))
    #
    #     print(len(signal_cut))
    #     print(len(sig))
    #
    #     ax2 = fig.add_subplot(212)
    #     ax2.set_title('Cut Signal ' + sample_file_name)
    #     ax2.set_ylabel('Amplitude')
    #     ax2.plot(signal_cut)
    #
    #     signal_cut = np.array(signal_cut)
    #
    #     mfcc_feat = mfcc(signal_cut, rate)
    #     mfcc_padded = self.padding(mfcc_feat)
    #
    #     print(mfcc_feat)
    #     print(mfcc_padded)
    #
    #     print(np.shape(mfcc_feat))
    #     print(np.shape(mfcc_padded))
    #
    #
    #     plt.show()
    #
    #     return 0

if __name__ == '__main__':
    fe = FeatureExtractor()
    fe.extract_features('./data/train/audio/', './data/train/features/')
    # fe.visualize('./data/train/audio/one/')