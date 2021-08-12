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


work_dir = '/media/dn/85E803050C839C68/m_fusion_data/'


def print_hi(name):
    print(f'Hi, {name}')
    DATA_PATH_JSON = work_dir + "data/sarcasm_data.json"
    BERT_TARGET_EMBEDDINGS = work_dir + "data/bert-output.jsonl"
    INDICES_FILE_OUR_INDEPENDENT_SPLIT = work_dir + "data/split_indices_oursplit_independent.p"
    AUDIO_PICKLE = work_dir + "data/audio_features.p"
    DATASET_FILE = work_dir + "mustard_oursplit.pkl"

    def pickle_loader(filename):
        if sys.version_info[0] < 3:
            return pickle.load(open(filename, 'rb'))
        else:
            return pickle.load(open(filename, 'rb'), encoding="latin1")

    def get_data_loader(train_ind_SI, author_ind):
        # (text,video,AUDIO)
        train_input = [data_input[ind] for ind in train_ind_SI]
        return train_input

    dataset_json = json.load(open(DATA_PATH_JSON))
    # text
    text_bert_embeddings = []
    with jsonlines.open(BERT_TARGET_EMBEDDINGS) as reader:
        print('opend bert : ', BERT_TARGET_EMBEDDINGS)
        for obj in reader:
            CLS_TOKEN_INDEX = 0
            features = obj['features'][CLS_TOKEN_INDEX]
            bert_embedding_target = []
            for layer in [0, 1, 2, 3]:
                bert_embedding_target.append(np.array(features["layers"][layer]["values"]))
            bert_embedding_target = np.mean(bert_embedding_target, axis=0)
            # text_bert_embeddings (list:690)    768
            text_bert_embeddings.append(np.copy(bert_embedding_target))
    print('np.array(text_bert_embeddings).shape bert 768 ')
    print(np.array(text_bert_embeddings).shape)  # 690 768
    # video
    video_features_file = h5py.File(work_dir + 'data/features/utterances_final/resnet_pool5.hdf5')
    # combined feature index
    # audio dict (283 12)   (283 11)
    audio_features = pickle_loader(AUDIO_PICKLE)

    # parse_data
    data_input, data_output = [], []
    # from nltk.tokenize import word_tokenize

    # data = "All work and no play makes jack a dull boy, all work and no play"
    # print(word_tokenize(data))
    # data_input [(text,video)(text,video)]
    # text:768 vide0: frame:2048
    for idx, ID in enumerate(dataset_json.keys()):
        text = dataset_json[ID]["utterance"]
        # text = word_tokenize(text)
        # len_text = len(text)

        video = video_features_file[ID][()] #(96,2048)(72,2048)
        # Zero in the end, deform, match the length of the sentence
        # len_video = video.shape[0]
        # dim_video = video.shape[1]
        # if len_video % len_text !=0 :
        #     len_pad = len_text - (len_video % len_text)
        #     pad_video = np.zeros((len_pad,dim_video))
        #     video = np.concatenate((video,pad_video))
        #
        # video = video.reshape(len_text,-1,dim_video)
        # video = np.mean(video,axis=1)

        audio = audio_features[ID] #(283,12) (283,11)
        audio = audio.T
        # # Zero in the end, deform, match the length of the sentence
        # len_audio = audio.shape[0]
        # dim_audio = audio.shape[1]
        # if len_audio % len_text != 0:
        #     len_pad = len_text - (len_audio % len_text)
        #     pad_audio = np.zeros((len_pad, dim_audio))
        #     audio = np.concatenate((audio, pad_audio))
        #
        # audio = audio.reshape(len_text, -1, dim_audio)
        # audio = np.mean(audio, axis=1)

        label = int(dataset_json[ID]["sarcasm"])
        label = 1.0 if label else -1.0
        label = np.array([[label]])
        segment = ID
        # (words, visual, acoustic), label_id, segment,
        data_input.append((
            (text,  # 0 TEXT_ID
             video,  # 1 VIDEO_ID
             audio,           # 2
             ),
            label,
            segment)
        )
        data_output.append(int(dataset_json[ID]["sarcasm"]))

    video_features_file.close()

    split_indices = pickle_loader(INDICES_FILE_OUR_INDEPENDENT_SPLIT)
    # 对于独立于说话者的设置 5组索引是相同的。
    for fold, (train_index, test_index) in enumerate(split_indices[:1]):
        train_ind_SI = train_index
        val_ind_SI = test_index
        test_ind_SI = test_index

        train_dataLoader = get_data_loader(train_ind_SI,None)
        val_dataLoader = get_data_loader(val_ind_SI,None)
        test_dataLoader = get_data_loader(test_ind_SI,None)

        data = {}
        data["train"] = train_dataLoader
        data["dev"] = val_dataLoader
        data["test"] = test_dataLoader

        if not os.path.exists(DATASET_FILE):
            pickle.dump(data, open(DATASET_FILE, 'wb'), protocol=2)

        print(f"write to {DATASET_FILE} ...")


if __name__ == '__main__':
    print_hi('lf_dnn')
