import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
file_4 = [ 'M', 'T', 'A', 'V']
labels = {}
label_score = {}
for file in file_4:
    labels[file] = pd.read_csv(os.path.join(label_dir, f'label_{file}.csv'))
    label_score[file] = labels[file].label
print(type(label_score['M']))
print(len(label_score['M']))
print(label_score['M'][:20])
label_M = label_score['M']
label_T = label_score['T']
label_A = label_score['A']
label_V = label_score['V']

print(label_M.value_counts())
print(label_M.value_counts()[0])
print(label_M.value_counts()[1])
print(label_M.value_counts()[-0.2])
print(type(label_M.value_counts()))
print(type(label_M))
print(len(label_M))

NG = []
WN = []
NU = []
WP = []
PS = []
for i in tqdm(range(len(label_M))):

    if label_M[i] == -1.0 or label_M[i] == -0.8:
        # print(i,label_M[i])
        NG.append(i)
    if label_M[i] == -0.6 or label_M[i] == -0.4 or label_M[i] == -0.2 :
        # print(i,label_M[i])
        WN.append(i)

    if label_M[i] == 0 :
        # print(i,label_M[i])
        NU.append(i)

    if label_M[i] == 0.6 or label_M[i] == 0.4 or label_M[i] == 0.2 :
        # print(i,label_M[i])
        WP.append(i)
    if label_M[i] == 1.0 or label_M[i] == 0.8:
        # print(i,label_M[i])
        PS.append(i)

print(len(NG))
print('NG[:10]')
print(NG[:10])

from sklearn.model_selection import train_test_split  #数据分区
# x=np.arange(72).reshape(24,3)  #24个样本点，3个维度
# y=np.arange(754)

def split_train(NG):
    X_train,X_test=train_test_split(NG,test_size=0.2,random_state=0)
    # X_train,X_test,y_train,y_test=train_test_split(NG,y,test_size=0.2,random_state=0)


    print(len(X_train),len(X_test))
    # print(len(X_train),len(X_test),len(y_train),len(y_test))
    print('X_train[:10]')
    print(X_train[:10])
    x_true_train ,X_valid =train_test_split(X_train,test_size=1/4,random_state=0)
    print(len(x_true_train),len(X_valid),len(X_test))
    print('x_true_train[:10]')
    print(x_true_train[:10])
    return x_true_train,X_valid,X_test


total = [NG,WN,NU,WP,PS]
trains =[]
vals = []
tests = []
for i in range(len(total)):
    train , val , xtest = split_train(total[i])
    trains.extend(train)
    vals.extend(val)
    tests.extend(xtest)

print('len(trains),len(vals),len(tests)')
print(len(trains),len(vals),len(tests))


print(trains[:10])
print(vals[:10])
print(tests[:10])


print(type(trains))