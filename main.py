from feature_extractor import *

fe = FeatureExtractor("./data/train/audio/one/", ",")

print(fe.sample(10))
# fe.visualize()
