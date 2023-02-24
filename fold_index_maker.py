import os
from utils import get_train_val_lists
import pandas as pd
import numpy as np

file_ptr = open('data/new_mapping_gestures.txt', 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])



image_path = '/datashare/APAS/frames/'

for fold_num in range(5):
    train, val, test = get_train_val_lists(fold_num, 'data/folds', f'/datashare/APAS/features/fold{fold_num}/')
    image_paths = []
    labels = []
    for video_name in train:
        image_names = sorted(os.listdir(image_path+video_name+'_side/'))
        file_ptr = open(f'data/gestures_modified/{video_name}.txt', 'r')
        gt_file_content = file_ptr.read().split('\n')[:-1]
        content = []
        for item in gt_file_content:
            splitted_item = item.split()
            content += [splitted_item[2]]*(int(splitted_item[1]) - int(splitted_item[0]) + 1)
        classes = [0]*min(len(image_names), len(content))
        for i in range(len(classes)):
            classes[i] = actions_dict[content[i]]
        if not(classes) or not(image_names):
            print('zift')
        classes = classes[:min(len(image_names), len(content))]
        image_names = image_names[:min(len(image_names), len(content))]
        image_paths += [os.path.join(image_path, video_name+'_side', im_name) for im_name in image_names]
        labels += classes
    dataset = pd.DataFrame(dict(image_path=image_paths, label=labels))
    dataset.to_csv(f'fold_indexes/fold{fold_num}_train.csv', index=False)
    image_paths = []
    labels = []
    for video_name in val:
        image_names = sorted(os.listdir(image_path+video_name+'_side/'))
        file_ptr = open(f'data/gestures_modified/{video_name}.txt', 'r')
        gt_file_content = file_ptr.read().split('\n')[:-1]
        content = []
        for item in gt_file_content:
            splitted_item = item.split()
            content += [splitted_item[2]]*(int(splitted_item[1]) - int(splitted_item[0]) + 1)
        classes = [0]*min(len(image_names), len(content))
        for i in range(len(classes)):
            classes[i] = actions_dict[content[i]]
        if not(classes) or not(image_names):
            print('zift')
        classes = classes[:min(len(image_names), len(content))]
        image_names = image_names[:min(len(image_names), len(content))]
        image_paths += [os.path.join(image_path, video_name+'_side', im_name) for im_name in image_names]
        labels += classes
    dataset = pd.DataFrame(dict(image_path=image_paths, label=labels))
    dataset.to_csv(f'fold_indexes/fold{fold_num}_val.csv', index=False)

