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
    label_score[file] = labels[file].label.tolist()
print(type(label_score['M']))
print(len(label_score['M']))
print(label_score['M'][:20])
label_M = label_score['M']
label_T = label_score['T']
label_A = label_score['A']
label_V = label_score['V']
tmp = np.array(label_M) - np.array(label_T)
print(tmp)
tmp = np.square(tmp)
print(tmp)
tmp = np.sum(tmp) / len(label_M)
print(tmp)
tmp = np.round(tmp,2)
print(tmp)
cnf_matrix = np.zeros((4,4))
print(cnf_matrix)

def score(label_M,label_T):
    tmp = np.array(label_M) - np.array(label_T)
    print(tmp)
    tmp = np.square(tmp)
    print(tmp)
    tmp = np.sum(tmp) / len(label_M)
    print(tmp)
    tmp = np.round(tmp, 2)
    print(tmp)

    return tmp

label_4 = [label_M,label_T,label_A,label_V]

for i in range(4):
    for j in range(4):
        cnf_matrix[i,j] = score(label_4[i],label_4[j])

print(cnf_matrix)


import itertools
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges
                          # cmap=plt.cm.Blues
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    # fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.show()
    # plt.savefig('confusion_matrix',dpi=200)


# cnf_matrix = np.array([
#     [0.00, 0.25, 0.14, 0.17],
#     [0.25, 0.00, 0.28, 0.46,],
#     [0.14, 0.28, 0.00, 0.21,],
#     [0.17, 0.46, 0.21, 0.00, ],
# ])

class_names = ['M', 'T', 'A', 'V']

# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                      title=' ')






