"""
准备Mustard数据集，方便调用
"""
import os
import sys
import re
import json
import pickle

from tqdm import tqdm
import h5py
import nltk
import numpy as np
import jsonlines
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from torchsummary import summary
from torch import optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

work_dir = '/media/dn/85E803050C839C68/m_fusion_data/'


def get_testdataset_segments_labels():
    DATA_PATH_JSON = work_dir + "data/sarcasm_data.json"

    dataset_json = json.load(open(DATA_PATH_JSON))

    data_input = []

    for idx, ID in enumerate(dataset_json.keys()):
        text = dataset_json[ID]["utterance"]
        label = dataset_json[ID]["sarcasm"]
        segment = ID
        speaker = dataset_json[ID]['speaker']
        # (words, visual, acoustic), label_id, segment,
        data_input.append((
            text,
            label,
            segment,
            speaker
        )
        )

    labels = [1 if data[1] else -1 for data in data_input]
    index = 5
    label_dir = '/media/dn/85E803050C839C68/m_fusion_data/bert_mustard/metadata'
    test_index = np.array(pd.read_csv(os.path.join(label_dir, f'test_index{index}.csv'))).reshape(-1)
    labels_test = np.array(labels)[test_index]
    segments = [(data[2],data[1],data[0],data[3]) for data in data_input]

    test_segments_labels = np.array(segments)[test_index]

    return test_segments_labels
    # save to csv,to simple analysis the error predict
    # (base) dn@dn-3080IPASON:~$ ls /media/dn/85E803050C839C68/datasets/mmsd_raw_data/utterances_final/|wc -l
    # 690

if __name__ == '__main__':
    get_testdataset_segments_labels()
