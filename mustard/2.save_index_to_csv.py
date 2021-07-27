'''
save 5fold,independent,our independent split indext to csv file
author : dn
date : 2021年 07月 10日 星期六 13:35:22 CST
'''
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
from globals import M_FUSION_DATA_PATH 
from utils.functions import  pickle_loader

work_dir = M_FUSION_DATA_PATH 
INDICES_FILE_5_FOLD = work_dir + "data/split_indices.p"
INDICES_FILE_OUR_I  = work_dir + "data/split_indices_oursplit_independent.p"
INDICES_FILE_SRC_I  = work_dir + "data/split_indices_source_independent.p"

label_dir = work_dir + "bert_mustard/csv"
DATA_PATH_JSON = work_dir + "data/sarcasm_data.json"
dataset_json = json.load(open(DATA_PATH_JSON))

def gen_5_fold():
    fold_5 = pickle_loader(INDICES_FILE_5_FOLD)
    for index,split_indice in enumerate(fold_5):
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


def gen_our_i():
    split_indices = pickle_loader(INDICES_FILE_OUR_I)
    import pdb;pdb.set_trace()
    for index,split_indice in enumerate(split_indices[:1]):
        index += 5
        (train_index, test_index) = split_indice
        print(f'train,test:{train_index.shape} {test_index.shape}')
        data = {'index': train_index}
        data_df = pd.DataFrame(data)
        data_df.to_csv(os.path.join(label_dir, f'train_index{index}.csv'), index=False)
        data = {'index': test_index}
        data_df = pd.DataFrame(data)
        data_df.to_csv(os.path.join(label_dir, f'test_index{index}.csv'), index=False)
        print("write to " + os.path.join(label_dir, f'train_index{index}.csv'))
        print("write to " + os.path.join(label_dir, f'test_index{index}.csv'))

def getSpeakerIndependent():
    train_ind_SI, test_ind_SI = [], []
    import pdb;pdb.set_trace()
    for idx, ID in enumerate(dataset_json.keys()):
        if dataset_json[ID]['show'] == "FRIENDS":
            test_ind_SI.append(idx)
        else:
            train_ind_SI.append(idx)
    train_index, test_index = train_ind_SI, test_ind_SI
    return np.array(train_index), np.array(test_index)

def getSpeakerIndependent_ours():
    train_ind_SI, test_ind_SI = [], []
    test_speakers = ['HOWARD', 'SHELDON']
    for idx, ID in enumerate(dataset_json.keys()):
        speaker = dataset_json[ID]["speaker"]
        if speaker in test_speakers:
            test_ind_SI.append(idx)
        else:
            train_ind_SI.append(idx)

    train_index, test_index = train_ind_SI, test_ind_SI
    return np.array(train_index), np.array(test_index)

def gen_src_i():
    (train_index, test_index) = getSpeakerIndependent()
    index = 6
    print(f'train,test:{train_index.shape} {test_index.shape}')
    data = {'index': train_index}
    data_df = pd.DataFrame(data)
    data_df.to_csv(os.path.join(label_dir, f'train_index{index}.csv'), index=False)
    data = {'index': test_index}
    data_df = pd.DataFrame(data)
    data_df.to_csv(os.path.join(label_dir, f'test_index{index}.csv'), index=False)
    print("write to " + os.path.join(label_dir, f'train_index{index}.csv'))
    print("write to " + os.path.join(label_dir, f'test_index{index}.csv'))

if __name__ == '__main__':
    #gen_5_fold()
    #gen_our_i()
    gen_src_i()
