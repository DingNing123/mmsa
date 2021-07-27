"""
准备Mustard数据集，方便调用
"""
import os
import sys
import re
import json
import pickle

import librosa
import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import jsonlines
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchsummary import summary
from torch import optim
from sklearn.metrics import classification_report, confusion_matrix
from globals import M_FUSION_DATA_PATH ,raw_data_path
from utils.functions import  pickle_loader

work_dir = M_FUSION_DATA_PATH 
DATA_PATH_JSON = work_dir + "data/sarcasm_data.json"
AUDIO_PICKLE = work_dir + "data/audio_features.p"
DATASET_FILE_FINAL = work_dir + "mustard_33_final.pkl"
DATASET_FILE_CONTEXT = work_dir + "mustard_33_context.pkl"
final_video_path = raw_data_path + 'utterances_final/'
context_video_path = raw_data_path + 'context_final/'    
final_audio_path = raw_data_path + 'Processed/audio/final/'
context_audio_path = raw_data_path + 'Processed/audio/context/'
final_pkl_file = work_dir + 'data/audio_final_33.pkl'
context_pkl_file = work_dir + 'data/audio_context_33.pkl'

dataset_json = json.load(open(DATA_PATH_JSON))

def gen_mustard_pkl():
    print(f'gen_mustard_pkl')
    # video
    video_features_file = h5py.File(work_dir + 'data/features/utterances_final/resnet_pool5.hdf5')
    # audio dict (283 12)   (283 11)
    # audio_features = pickle_loader(AUDIO_PICKLE)
    with open(final_pkl_file, "rb") as handle:
        audio_features = pickle.load(handle)

    #import pdb;pdb.set_trace()

    # parse_data
    data_input = []
    #for idx, ID in enumerate(list(dataset_json.keys())[:2]):
    for idx, ID in enumerate(tqdm(dataset_json.keys())):
        text = dataset_json[ID]["utterance"]
        # maybe there is not only one person such as ['LEONARD', 'SHELDON']
        # I think [0 0 0 1 0 1] for 1 is some one persion,when using one hot vector
        # 这里video的序列长度太长了。我觉得可以 Zoom 10 times，缩放10倍
        # 先对10取余，余下的求平均。
        video = video_features_file[ID][()] #(96,2048)(72,2048)
        right = video.shape[0] % 10 #6
        left = video.shape[0] - right
        video = np.concatenate(
            (np.mean(np.reshape(video[:left], (-1, 10, 2048)), axis=1), np.expand_dims(np.mean(video[-left:], axis=0), axis=0)),
            axis=0)

        audio = audio_features[ID] #(283,12) (283,11)
        #audio = audio.T

        label = int(dataset_json[ID]["sarcasm"])
        label = 1.0 if label else -1.0
        label = np.array([[label]])
        segment = ID
        data_input.append((
            (text,  
             video, 
             audio, 
             ),
            label,
            segment)
        )

    video_features_file.close()
    data = data_input

    #if not os.path.exists(DATASET_FILE):
    pickle.dump(data, open(DATASET_FILE_FINAL, 'wb'), protocol=2)
    print(f"saved to {DATASET_FILE_FINAL} ...")

def gen_mustard_pkl_context():
    dataset_json = json.load(open(DATA_PATH_JSON))
    # video
    video_final_features_file = h5py.File(work_dir + 'data/features/utterances_final/resnet_pool5.hdf5')
    video_context_features_file = h5py.File(work_dir + 'data/features/context_final/resnet_pool5.hdf5')
    with open(final_pkl_file, "rb") as handle:
        audio_final_features = pickle.load(handle)
    with open(context_pkl_file, "rb") as handle:
        audio_context_features = pickle.load(handle)

    # parse_data
    data_input = []
    #for idx, ID in enumerate(list(dataset_json.keys())[:9]):
    for idx, ID in enumerate(tqdm(dataset_json.keys())):
        text_final = dataset_json[ID]["utterance"]
        text_context = dataset_json[ID]['context'] 
        speaker_context = dataset_json[ID]['context_speakers']
        speaker_final = dataset_json[ID]['speaker']
        # maybe there is not only one person such as ['LEONARD', 'SHELDON']
        # I think [0 0 0 1 0 1] for 1 is some one persion,when using one hot vector
        # 这里video的序列长度太长了。我觉得可以 Zoom 10 times，缩放10倍
        # 先对10取余，余下的求平均。
        video = video_final_features_file[ID][()] #(96,2048)(72,2048)
        right = video.shape[0] % 10 #6
        left = video.shape[0] - right
        video_final = np.concatenate(
            (np.mean(np.reshape(video[:left], (-1, 10, 2048)), axis=1), np.expand_dims(np.mean(video[-left:], axis=0), axis=0)),
            axis=0)
        #####video context feature handle
        video_context = video_context_features_file[ID][()] #(96,2048)(72,2048)
        right = video_context.shape[0] % 10 #6
        left =  video_context.shape[0] - right
        video_context = np.concatenate(
            (np.mean(np.reshape(video_context[:left], (-1, 10, 2048)), axis=1), np.expand_dims(np.mean(video_context[-left:], axis=0), axis=0)),
            axis=0)
        ##############################

        audio_final = audio_final_features[ID] #(283,12) (283,11)
        audio_context = audio_context_features[ID] #(283,12) (283,11)

        label = int(dataset_json[ID]["sarcasm"])
        label = 1.0 if label else -1.0
        label = np.array([[label]])
        segment = ID
        data_input.append((
            (text_final,  
             video_final, 
             audio_final, 
             ),
            text_context ,
            video_context,
            audio_context,
            speaker_final,
            speaker_context,
            label,
            segment
            ))

    video_final_features_file.close()
    video_context_features_file.close()
    data = data_input

    pickle.dump(data, open(DATASET_FILE_CONTEXT, 'wb'), protocol=2)
    print(f"saved to {DATASET_FILE_CONTEXT} ...")

def video_to_audio(video_path,audio_path):
    video_paths = sorted(glob(os.path.join(video_path,'*.mp4')))
    for vp in tqdm(video_paths[:]):
        output_path = vp.replace(video_path,audio_path ).replace('.mp4','.wav')
        cmd = 'ffmpeg -i ' + vp + ' -f wav -vn ' + output_path + ' -loglevel quiet'
        os.system(cmd)


def gen_audios():
    '''fetch audios mp4  -> wav
    '''
    video_to_audio(final_video_path,final_audio_path)
    video_to_audio(context_video_path,context_audio_path)

def gen_final_audio_pkl():
    audio_dict = {}
    lengths = []
    #for idx, ID in enumerate(list(dataset_json.keys())[:9]):
    for idx, ID in enumerate(tqdm(dataset_json.keys())):
        audio_path = final_audio_path + ID + '.wav'
        y, sr = librosa.load(audio_path)
        hop_length = 5120
        f0 = librosa.feature.zero_crossing_rate(y,hop_length=hop_length).T # (seq,1)
        mfcc = librosa.feature.mfcc(y=y,hop_length=hop_length,htk=True).T # (seq,20)
        #mfcc_delta = librosa.feature.delta(mfcc.T).T # seq,20
        cqt = librosa.feature.chroma_cqt(y=y,sr=sr,hop_length=hop_length).T #(seq,12)
        #mel = librosa.feature.melspectrogram(y=y,sr=sr,hop_length=hop_length).T # seq,128

        audio_33 = np.concatenate([f0,mfcc,cqt], axis=-1)
        audio_dict[ID] = audio_33
        length = audio_33.shape[0]
        lengths.append(length)

    lengths = np.array(lengths)
    print(f'mean:{lengths.mean():.4f}, std: {lengths.std():.4f}')

    with open (final_pkl_file ,'wb') as f:
        pickle.dump(audio_dict,f,pickle.HIGHEST_PROTOCOL)
        print(f'saved in {final_pkl_file}')

def gen_context_audio_pkl():
    audio_dict = {}
    lengths = []
    # import pdb;pdb.set_trace()
    #for idx, ID in enumerate(list(dataset_json.keys())[:9]):
    for idx, ID in enumerate(tqdm(dataset_json.keys())):
        audio_path = context_audio_path + ID + '_c.wav'
        y, sr = librosa.load(audio_path)
        hop_length = 5120
        f0 = librosa.feature.zero_crossing_rate(y,hop_length=hop_length).T # (seq,1)
        mfcc = librosa.feature.mfcc(y=y,hop_length=hop_length,htk=True).T # (seq,20)
        #mfcc_delta = librosa.feature.delta(mfcc.T).T # seq,20
        cqt = librosa.feature.chroma_cqt(y=y,sr=sr,hop_length=hop_length).T #(seq,12)
        #mel = librosa.feature.melspectrogram(y=y,sr=sr,hop_length=hop_length).T # seq,128

        audio_33 = np.concatenate([f0,mfcc,cqt], axis=-1)
        audio_dict[ID] = audio_33
        length = audio_33.shape[0]
        lengths.append(length)

    lengths = np.array(lengths)
    print(f'mean:{lengths.mean():.4f}, std: {lengths.std():.4f}')

    with open (context_pkl_file ,'wb') as f:
        pickle.dump(audio_dict,f,pickle.HIGHEST_PROTOCOL)
        print(f'saved in {context_pkl_file}')

def read_context_audio_pkl():
    path = context_pkl_file
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    print(data.keys())

if __name__ == '__main__':
    # gen_mustard_pkl()
    gen_mustard_pkl_context()
    # gen_audios()
    #gen_final_audio_pkl()
    #gen_context_audio_pkl()
    # read_context_audio_pkl()
