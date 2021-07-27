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
print( type(label_M.value_counts()))

def count_sentiment(label_M):
    tmp = label_M.value_counts()
    x1 = tmp[-1.0] + tmp[-0.8]
    x2 = tmp[-0.6] + tmp[-0.4] + tmp[-0.2]
    x3 = tmp[0]
    x4 = tmp[0.6] + tmp[0.4] + tmp[0.2]
    x5 = tmp[1.0] + tmp[0.8]
    tmp = [x1,x2,x3,x4,x5]
    return tmp

M = count_sentiment(label_M)
women_means = count_sentiment(label_T)
women_3 = count_sentiment(label_A)
women_4 = count_sentiment(label_V)


print()
print('M--------------')
print(M)
print(np.round(np.array(M)*0.6),0)
print(np.round(np.array(M)*0.2),0)




labels = ['Negative', 'Weakly Negative', 'Neutral', 'Weakly Positive', 'Positive']
# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]
# women_3 = [25, 32, 34, 20, 25]
# women_4 = [25, 32, 34, 20, 25]
x = np.arange(len(labels))  # the label locations
x = x * 2
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width*3/2, M, width, label='M')
rects2 = ax.bar(x - width/2, women_means, width, label='T')
rects3 = ax.bar(x + width/2, women_3, width, label='A')
rects4 = ax.bar(x + width*3/2, women_4, width, label='V')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
# plt.ylim(0,50)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)

fig.tight_layout()

plt.show()