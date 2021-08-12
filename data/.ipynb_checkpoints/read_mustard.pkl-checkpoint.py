import os
import sys
import re
import json
import pickle


def pickle_loader(filename):
    if sys.version_info[0] < 3:
        return pickle.load(open(filename, 'rb'))
    else:
        return pickle.load(open(filename, 'rb'), encoding="latin1")


work_dir = '/media/dn/85E803050C839C68/m_fusion_data/'
DATASET_FILE = work_dir + "mustard.pkl"
mustard = pickle_loader(DATASET_FILE)
print(type(mustard))