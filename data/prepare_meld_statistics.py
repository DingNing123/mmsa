import os
import argparse
import librosa
import struct
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import pickle

import torch
from transformers import *

data_dir = "/media/dn/85E803050C839C68/m_fusion_data/bert_meld/"


with open(os.path.join(data_dir,"meld.pkl"), "rb") as handle:
    data = pickle.load(handle)


# dict_keys(['train', 'valid', 'test'])
# dict_keys(['text', 'audio', 'labels', 'segments'])

# data['train']['text'].shape
# (9989, 18, 768)
# data['train']['audio'].shape
# (9989, 1, 600)


# data['train']['text'].shape
# (9989, 18, 768)
# data['valid']['text'].shape
# (1109, 17, 768)
# data['test']['text'].shape
# (2610, 18, 768)

print()