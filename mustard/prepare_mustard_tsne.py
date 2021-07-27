import numpy as np

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import seaborn as sns
import os

def save_plot(save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    fig = plt.gcf()

    fig.savefig(save_path, dpi=300)
    print('png saved in: ', save_path)

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright",2)

modal = 'Feature_All'
work_dir = '/media/dn/85E803050C839C68/m_fusion_data/'
path =  work_dir + f'representation/{modal}.npz'
data = np.load(path)
class_name=['non-sarcastic','sarcastic']
# res1 = {
#                     'repre_f': repre_f,
#                     'repre_t': repre_t,
#                     'repre_a': repre_a,
#                     'repre_v': repre_v,
#                     'label': true_numpy
#                 }
title = 'repre_f'
# title = 'repre_t'
# title = 'repre_a'
# title = 'repre_v'

X = data[title]
y4 = data['label']
y4 = y4.squeeze(axis=1)
y4 = [1 if y==1 else 0 for y in y4]
y4 = [class_name[yi] for yi in y4]
tsne = TSNE()
X_embedded = tsne.fit_transform(X)
# print(y4)
print(X_embedded.shape)
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y4, legend='full', palette=palette)
# plt.title(modal)
plt.title(title,fontsize=30)
plt.legend(fontsize=30)
path = work_dir + f'representation/img/{title}.png'
save_plot(path)

os.system("shotwell " + path)

# path = work_dir + f'representation/img/{modal}.eps'
# save_plot(path)
