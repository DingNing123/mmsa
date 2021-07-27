import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)
x = np.array([0,1,2,3,4,0,0,0,0,1,1,1,2,2,3])

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



# the histogram of the data
n, bins, patches = plt.hist(label_M, bins = [-1,-0.6,-0.1,0.1,0.7,1], alpha=0.75,label='M')
n, bins, patches = plt.hist(label_T, bins = [-1,-0.6,-0.1,0.1,0.7,1], alpha=0.75,label='T')
# n, bins, patches = plt.hist(x, 5, density=True, facecolor='g', alpha=0.75)
# print('n,bins,patches')
# print(n,bins,patches)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.xlim(40, 160)
# plt.ylim(0, 0.03)
# plt.grid(True)
plt.legend()
plt.show()