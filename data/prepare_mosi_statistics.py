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

data_dir = "/media/dn/85E803050C839C68/m_fusion_data/bert_mosi/"


with open(os.path.join(data_dir,"mosi.pkl"), "rb") as handle:
    data = pickle.load(handle)


# dict_keys(['train', 'valid', 'test'])
# dict_keys(['vision', 'text', 'audio', 'labels', 'segments'])

# data['train']['text'].shape
# (1281, 21, 768)
# data['train']['audio'].shape
# (1281, 20, 74)
# data['train']['vision'].shape
# (1281, 20, 47)

# data['train']['text'].shape
# (1281, 21, 768)
# data['valid']['text'].shape
# (229, 20, 768)
# data['test']['text'].shape
# (685, 26, 768)

print()