import argparse
import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import os
from globals import M_FUSION_DATA_PATH 


def save_plot(save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    fig = plt.gcf()
    fig.savefig(save_path, dpi=300)
    print('png saved in: ', save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='audio',
                        help='audio/text/visual/tva')
    return parser.parse_args()

def get_axs(modal,ax):
    path =  work_dir + f'representation/{modal}.npz'
    data = np.load(path)
    # non-sarcstic :N sarcastic: Y
    class_name=['N','Y']
    X = data['repre']
    y4 = data['label']
    y4 = [class_name[yi] for yi in y4]

    X_embedded = tsne.fit_transform(X)
    return sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y4, legend='full',
            palette=palette,ax = ax)


if __name__ == '__main__':
    # args = parse_args()
    work_dir = M_FUSION_DATA_PATH 
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    palette = sns.color_palette("bright",2)
    tsne = TSNE()
    fig, axs = plt.subplots(2, 2)

    get_axs('text',axs[0,0])
    axs[0,1] = get_axs('audio',axs[0,1])
    axs[1,0] = get_axs('visual',axs[1,0])
    axs[1,1] = get_axs('tva',axs[1,1])

    axs[0,0] .set_title('text'.title(),fontsize=20)
    axs[0,1] .set_title('audio'.title(),fontsize=20)
    axs[1,0] .set_title('visual'.title(),fontsize=20)
    axs[1,1] .set_title('tva'.title(),fontsize=20)

    # axs[0,0] .legend(fontsize=20)
    plt.legend()
    # plt.show()
    path =  work_dir + f'representation/img/all.png'
    save_plot(path)
    #path =  work_dir + f'representation/img/{modal}.eps'
    #save_plot(path)
