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

data_dir = "/media/dn/85E803050C839C68/m_fusion_data/bert_mustard/"


with open(os.path.join(data_dir,"mustard.pkl"), "rb") as handle:
    data = pickle.load(handle)

# data['vision'].shape
# (690, 21, 2048)
# data['text'].shape
# (690, 29, 768)
# data['audio'].shape
# (690, 12, 283)



print()