import random
from os import walk
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav

class FeatureExtractor:
    def __init__(self, path, sep=','):
        self.sample_names = []
        self.labels = []
        self.path = path

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
            mfcc_feat = mfcc(sig, rate)
            print(mfcc_feat, np.shape(mfcc_feat))
        return 0
