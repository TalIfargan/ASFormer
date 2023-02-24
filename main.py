import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval
from utils import get_train_val_lists
import os
import argparse
import numpy as np
import random
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--hidden', default='1280')
parser.add_argument('--dataset', default="50salads")
parser.add_argument('--split', default='0')
parser.add_argument('--model_dir', default='models_v2_hidden_1280')
parser.add_argument('--results_dir', default='results_v2')
parser.add_argument('--features_path', default='new_features')
parser.add_argument('--smooth', type=int, default=0)

args = parser.parse_args()


num_epochs = 6

lr = 0.0005
num_layers = 10
num_f_maps = 64
features_dim = 1280
bz = 1

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
# if args.dataset == "50salads":
#     sample_rate = 2
#
# # To prevent over-fitting for GTEA. Early stopping & large dropout rate
# if args.dataset == "gtea":
#     channel_mask_rate = 0.5
#
# if args.dataset == 'breakfast':
#     lr = 0.0001

# features_path = f'/datashare/APAS/features/fold{args.split}/'
features_path = os.path.join(args.features_path, f'hidden_{args.hidden}', f'fold{args.split}')
train_list, val_list, test_list = get_train_val_lists(args.split, os.path.join('data', 'folds'), features_path)

gt_path = os.path.join('data', 'transcriptions_gestures')
 
mapping_file = os.path.join('data', 'mapping_gestures.txt')
 
model_dir = os.path.join(f'models_v2_hidden_{args.hidden}', f'split_{args.split}')

if args.smooth:
    model_dir = os.path.join(f'models_smooth_{args.smooth}', f'split_{args.split}')

results_dir = os.path.join(args.results_dir+f'_hidden_{args.hidden}', f'split_{args.split}')

if args.smooth:
    results_dir = os.path.join(f'results_smooth_{args.smooth}', f'split_{args.split}')
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
 
 
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)

trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate, smooth=args.smooth)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(train_list)

    batch_gen_val = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_val.read_data(val_list)

    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_val, args.split)

if args.action == "predict":
    batch_gen_test = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_test.read_data(test_list)
    trainer.predict(model_dir, results_dir, features_path, batch_gen_test, actions_dict, sample_rate)

