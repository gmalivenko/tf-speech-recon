import random
from os import walk
import numpy as np
import array
import struct

from python_speech_features import mfcc
import webrtcvad
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from wav_crop import crop_wav

# from pydub import AudioSegment,silence



class FeatureExtractor:
    def __init__(self, path, sep=','):
        self.sample_names = []
        self.labels = []
        self.path = path

        self.vad = webrtcvad.Vad()


        for (dirpath, dirnames, filenames) in walk(path):
            self.sample_names.extend(filenames)
            break

        # with open(filename, 'r') as f:
        #     next(f) #skip first line
        #     i = 1
        #     for line in f:
        #         index_tuple = line.split(sep=sep)
        #         if i == 1:
        #             print(index_tuple)
        #             i +=1
        #
        #         self.sample_names.append(index_tuple[0])
        #         self.labels.append(index_tuple[1])

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

            mfcc_feat = mfcc(signal_cut, rate)

        return mfcc_feat


    def padding(self):
        return 0

    def visualize(self):
        id = random.sample(range(len(self.sample_names)), 1)[0]
        sample_file_name = self.path + self.sample_names[id]
        # sample_file_name = self.path + "0f7dc557_nohash_2.wav"
        (rate, sig) = wav.read(sample_file_name)

        print(sample_file_name)

        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(211)
        ax1.set_title('Raw wave of ' + sample_file_name)
        ax1.set_ylabel('Amplitude')
        ax1.plot(sig)

        signal_cut = []
        # mfcc_feat = mfcc(sig, rate)
        raw_cut = crop_wav(sig, rate)

        for i,s in enumerate(raw_cut):
            print(s)
            print(array.array("h", s))
            signal_cut.extend(array.array("h", s))

        print(len(signal_cut))
        print(len(sig))

        ax2 = fig.add_subplot(212)
        ax2.set_title('Cut Signal ' + sample_file_name)
        ax2.set_ylabel('Amplitude')
        ax2.plot(signal_cut)

        plt.show()

        return 0
