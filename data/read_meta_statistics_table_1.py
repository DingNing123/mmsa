import os
import argparse
import librosa
import struct
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

data_dir = './CH-SIMS/'
meta_dir = './CH-SIMS/metadata/'
label_dir = meta_dir + 'sentiment/'
meta = pd.read_csv(os.path.join(meta_dir, 'meta.csv'))
print('type(meta)')
print(type(meta))
print('meta')
print(meta)
print(len(meta))
print(type(meta['age']))
print('female : 1')
print(meta.sex.sum())
print('male: 0 ')
print(len(meta.sex) - meta.sex.sum())
print('meta.sex.value_counts():   分类统计 计数 ' )
print(meta.sex.value_counts())
print('meta.video_id.value_counts() ')
print(meta.video_id.value_counts())
print('meta.video_id.nunique()   不重复统计  ')
print(meta.video_id.nunique())

print('meta.distinctive.value_counts():   分类统计 计数 ' )
print(meta.distinctive.value_counts())

label_t = pd.read_csv(os.path.join(label_dir, 'label_T.csv'))
print(label_t)
print(label_t.text)

total_lenth = 0
for text in tqdm(label_t.text):
    total_lenth += len(text)

print('average: Average word count per segments')
print(total_lenth/len(label_t.text))



