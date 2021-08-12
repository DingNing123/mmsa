import json
import os
import pickle
import sys
import numpy as np
import pandas as pd


def pickle_loader(filename):
    if sys.version_info[0] < 3:
        return pickle.load(open(filename, 'rb'))
    else:
        return pickle.load(open(filename, 'rb'), encoding="latin1")


work_dir = '/media/dn/85E803050C839C68/m_fusion_data/'
INDICES_FILE = work_dir + "data/split_indices_oursplit_independent.p"
label_dir = work_dir + "bert_mustard/metadata"

split_indices = pickle_loader(INDICES_FILE)
for index,split_indice in enumerate(split_indices[:1]):
    index += 5
    (train_index, test_index) = split_indice
    print(index,train_index[:15])
    data = {'index': train_index}
    data_df = pd.DataFrame(data)
    data_df.to_csv(os.path.join(label_dir, f'train_index{index}.csv'), index=False)
    data = {'index': test_index}
    data_df = pd.DataFrame(data)
    data_df.to_csv(os.path.join(label_dir, f'test_index{index}.csv'), index=False)
    print("write to " + os.path.join(label_dir, f'train_index{index}.csv'))
    print("write to " + os.path.join(label_dir, f'test_index{index}.csv'))

