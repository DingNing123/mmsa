import os
import random
import pickle
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import pdb

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

__all__ = ['MMDataLoader','MMDataLoader1']
author_ind = None

def toOneHot(data, size=None):
    '''
    Returns one hot label version of data
    '''
    oneHotData = np.zeros((len(data), size))
    oneHotData[range(len(data)), data] = 1

    assert (np.array_equal(data, np.argmax(oneHotData, axis=1)))
    return oneHotData

class MMDataset(Dataset):
    def __init__(self, args, index, mode='train'):
        self.args = args
        self.mode = mode
        self.index = index

        DATA_MAP = {
            'mosi': self.__init_mosi,
            'meld': self.__init_meld,
            'mustard': self.__init_mustard,
            'sims': self.__init_msaZH
        }
        DATA_MAP[args.datasetName](args)

    def __init_meld(self, args):
        with open(args.datapath, 'rb') as f:
            data = pickle.load(f)

        # now meld donnot have visual feature
        self.vision = data[self.mode]['audio'].astype(np.float32)

        self.text = data[self.mode]['text'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.label = {
            'M': np.array(data[self.mode]['labels']).astype(np.float32)
        }
        if 'need_normalize' in args.keys() and args.need_normalize:
            self.train_visual_max = np.max(np.max(np.abs(data['train']['audio']), axis=0), axis=0)
            self.train_visual_max[self.train_visual_max == 0] = 1
            self.__normalize()

    def __init_mosi(self, args):
        with open(args.datapath, 'rb') as f:
            data = pickle.load(f)

        self.vision = data[self.mode]['vision'].astype(np.float32)

        self.text = data[self.mode]['text'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.label = {
            'M': np.array(data[self.mode]['labels']).astype(np.float32)
        }
        if 'need_normalize' in args.keys() and args.need_normalize:
            self.train_visual_max = np.max(np.max(np.abs(data['train']['vision']), axis=0), axis=0)
            self.train_visual_max[self.train_visual_max == 0] = 1
            self.__normalize()

    def __init_mustard(self, args):
        with open(args.datapath, 'rb') as f:
            data = pickle.load(f)

        self.text = data['text_final'][self.index[self.mode]]
        self.vision = data['video_final'][self.index[self.mode]]
        self.audio = data['audio_final'][self.index[self.mode]]
        self.audio[self.audio == -np.inf] = 0

        if self.args.context :
            self.tcontext = data['text_context'][self.index[self.mode]]
            # pdb.set_trace()
            # for context [CLS] and text [CLS]
            args.csindex = self.tcontext.shape[1]
            self.text = np.concatenate([self.tcontext,self.text],axis=1)
            self.tvision = data['video_context'][self.index[self.mode]]
            self.vision = np.concatenate([self.tvision,self.vision],axis=1)
            self.taudio = data['audio_context'][self.index[self.mode]]
            self.audio = np.concatenate([self.taudio,self.audio],axis=1)

        if self.args.speaker :
            all_speakers = np.array(data['speaker_final'])
            train_speakers = all_speakers[self.index[self.mode]]
            global author_ind
            # pdb.set_trace()
            UNK_AUTHOR_ID = author_ind["PERSON"]
            authors = [author_ind.get(author.strip(), UNK_AUTHOR_ID) for author in train_speakers]
            authors_feature = toOneHot(authors, len(author_ind))  # 18 speaker
            self.speaker = authors_feature 
            

        self.label = {
            'M': np.array(data['labels'])[self.index[self.mode]]
        }

        if 'need_normalize' in args.keys() and args.need_normalize:
            self.train_visual_max = np.max(np.max(np.abs(data['video_final'][self.index['train']]), axis=0), axis=0)
            self.train_visual_max[self.train_visual_max == 0] = 1
            self.__normalize()

    def __init_msaZH(self, args):
        data = np.load(args.datapath)
        # 'datapath': '/datasets/CH-SIMS/Processed/features/data.npz'
        self.vision = data['feature_V'][self.index[self.mode]]
        # (2281, 55, 709) 2281片段 55帧画面 709维特征
        # 1368 个训练样本
        self.audio = data['feature_A'][self.index[self.mode]]
        self.text = data['feature_T'][self.index[self.mode]]

        self.label = {
            'M': data['label_M'][self.index[self.mode]],
            'T': data['label_T'][self.index[self.mode]],
            'A': data['label_A'][self.index[self.mode]],
            'V': data['label_V'][self.index[self.mode]]
        }

        if 'need_normalize' in args.keys() and args.need_normalize:
            self.train_visual_max = np.max(np.max(np.abs(data['feature_V'][self.index['train']]), axis=0), axis=0)
            self.train_visual_max[self.train_visual_max == 0] = 1
            self.__normalize()

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        # return len(self.labels)
        return len(self.index[self.mode])

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        
        if self.args.speaker :
            sample = {
                'text': torch.Tensor(self.text[index]),
                'audio': torch.Tensor(self.audio[index]),
                'vision': torch.Tensor(self.vision[index]),
                'speaker': torch.Tensor(self.speaker[index]),
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.label.items()}
            }
        else:
            sample = {
                'text': torch.Tensor(self.text[index]),
                'audio': torch.Tensor(self.audio[index]),
                'vision': torch.Tensor(self.vision[index]),
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.label.items()}
            }

        return sample

def speaker_dict(args,index,mode):
    with open(args.datapath, 'rb') as f:
        data = pickle.load(f)
    all_speakers = np.array(data['speaker_final'])
    train_speakers = all_speakers[index['train']]
    author_list = set()
    author_list.add("PERSON")
    for author in train_speakers:
        author = author.strip()
        if "PERSON" not in author:  # PERSON3 PERSON1 all --> PERSON haha
            author_list.add(author)

    author_ind = {author: ind for ind, author in enumerate(author_list)}

    return author_ind


def MMDataLoader(args):
    if args.datasetName == 'mosi' :
        # 其实有1281个样本，但是剩余的一个因为扩展维度 难以处理。所以舍弃
        train_index = np.arange(1280)
        val_index = np.arange(229)
        test_index = np.arange(685)

    if args.datasetName == 'meld' :
        train_index = np.arange(9989)
        val_index = np.arange(1109)
        test_index = np.arange(2610)

    if args.datasetName == 'mustard':
        if args.split == 'dep':
            index = args.cur_time - 1
        if args.split == 'our_i':
            index = 5
        if args.split == 'src_i':
            index = 6
            
        test_index = np.array(pd.read_csv(os.path.join(args.label_dir, f'test_index{index}.csv'))).reshape(-1)
        train_index = np.array(pd.read_csv(os.path.join(args.label_dir, f'train_index{index}.csv'))).reshape(-1)
        val_index = np.array(pd.read_csv(os.path.join(args.label_dir, f'test_index{index}.csv'))).reshape(-1)

    if args.datasetName == 'sims' :
        test_index = np.array(pd.read_csv(os.path.join(args.label_dir, 'test_index.csv'))).reshape(-1)
        train_index = np.array(pd.read_csv(os.path.join(args.label_dir, 'train_index.csv'))).reshape(-1)
        val_index = np.array(pd.read_csv(os.path.join(args.label_dir, 'val_index.csv'))).reshape(-1)

    index = {
        'train': train_index,
        'valid': val_index,
        'test': test_index
    }

    # speaker index dict and one hot vector
    if args.speaker:
        global author_ind
        author_ind = speaker_dict(args,index=index,mode='train')
        # pdb.set_trace()
        args.speakers = len(author_ind)

    datasets = {
        'train': MMDataset(args, index=index, mode='train'),
        'valid': MMDataset(args, index=index, mode='valid'),
        'test': MMDataset(args, index=index, mode='test')
    }

    # because normalize change t,a,v sequence, update  args.input_lens
    if 'input_lens' in args.keys():
        args.input_lens = datasets['train'].get_seq_len()

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }

    return dataLoader


def MMDataLoader1(args):
    """
    为了检查测试集分类出错的样本，因此固定测试集的索引顺序
    重点修改 DataLoader 类
    数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。

    参数：

    dataset (Dataset) – 加载数据的数据集。
    batch_size (int, optional) – 每个batch加载多少个样本(默认: 1)。
    shuffle (bool, optional) – 设置为True时会在每个epoch重新打乱数据(默认: False).
    sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
    num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
    collate_fn (callable, optional) –
    pin_memory (bool, optional) –
    drop_last (bool, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。
    如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)
    :param args:
    :return:
    """
    if args.datasetName == 'mosi' :
        # 其实有1281个样本，但是剩余的一个因为扩展维度 难以处理。所以舍弃
        train_index = np.arange(1280)
        val_index = np.arange(229)
        test_index = np.arange(685)

    if args.datasetName == 'meld' :
        # # data['train']['text'].shape
        # # (9989, 18, 768)
        # # data['valid']['text'].shape
        # # (1109, 17, 768)
        # # data['test']['text'].shape
        # # (2610, 18, 768)
        train_index = np.arange(9989)
        val_index = np.arange(1109)
        test_index = np.arange(2610)

    if args.datasetName == 'mustard':
        index = args.cur_time - 1
        test_index = np.array(pd.read_csv(os.path.join(args.label_dir, f'test_index{index}.csv'))).reshape(-1)
        train_index = np.array(pd.read_csv(os.path.join(args.label_dir, f'train_index{index}.csv'))).reshape(-1)
        val_index = np.array(pd.read_csv(os.path.join(args.label_dir, f'test_index{index}.csv'))).reshape(-1)

    if args.datasetName == 'sims' :
        test_index = np.array(pd.read_csv(os.path.join(args.label_dir, 'test_index.csv'))).reshape(-1)
        train_index = np.array(pd.read_csv(os.path.join(args.label_dir, 'train_index.csv'))).reshape(-1)
        val_index = np.array(pd.read_csv(os.path.join(args.label_dir, 'val_index.csv'))).reshape(-1)

    index = {
        'train': train_index,
        'valid': val_index,
        'test': test_index
    }

    datasets = {
        'train': MMDataset(args, index=index, mode='train'),
        'valid': MMDataset(args, index=index, mode='valid'),
        'test': MMDataset(args, index=index, mode='test')
    }
    # because normalize change t,a,v sequence, update  args.input_lens
    if 'input_lens' in args.keys():
        args.input_lens = datasets['train'].get_seq_len()

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=False)
        for ds in datasets.keys()
    }

    return dataLoader
