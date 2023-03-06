from sklearn.decomposition import PCA
import os
import numpy as np
from utils import get_train_val_lists

import argparse


def make_pca(videos, output_size, fold_num):
    X_t = []
    #videos_to_read = [video for video in os.listdir(f'new_features/hidden_1280/fold{fold_num}/') if video in videos]
    for i, video in enumerate(videos):
        X_t.append(np.load(f'new_features/hidden_1280/fold{fold_num}/{video}.npy').T)
        print(i)
        # if i == 3:
        #     break
    X = np.concatenate(X_t)
    pca = PCA(n_components=output_size)
    print("start of fit")
    pca.fit(X)
    print('zift')
    pca.explain_variance_ratio_.cumsum()
    return pca

# for output_size in [512, 256]:
#     for split in range(5):
#         if (output_size==512 and split == 0) or (output_size==512 and split == 1):
#             continue 


parser = argparse.ArgumentParser()
parser.add_argument('--output_size', default='64')
parser.add_argument('--split', default='4')
args = parser.parse_args()
output_size = int(args.output_size)
split = args.split


print(f"starts split {split} in size {output_size}")
train_list, val_list, test_list = get_train_val_lists(split, os.path.join('data', 'folds'), f'new_features/hidden_1280/fold{split}')
pca = make_pca(train_list, output_size, split)

for file in os.listdir(f'new_features/hidden_1280/fold{split}/'):
    npy = np.load(f'new_features/hidden_1280/fold{split}/{file}').T
    transformed = pca.transform(npy).T
    np.save(f'new_features/hidden_{output_size}/fold{split}/{file}', transformed)
del pca
        




