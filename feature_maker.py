import numpy as np
import torch
import os
from torchvision.models import efficientnet_v2_s, efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
from torchvision.io import read_image
from torchvision.models import EfficientNet_V2_S_Weights
from data.dataset_maker_2 import CustomImageDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

transforms = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
model_path = 'efficient_models'
frames_path = '/datashare/APAS/frames'
video_folders = [item for item in os.listdir(frames_path) if item.endswith('side')]
save_path = 'new_features/hidden_1280/'
device = ('cuda' if torch.cuda.is_available() else 'cpu')

for fold_num in ['0', '1', '2', '3', '4']:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(1280, 18, bias=True)
    model.load_state_dict(torch.load(os.path.join(model_path, f'fold{fold_num}_model.pkl')))
    model.classifier = Identity()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for video in video_folders:
            video_save_path = os.path.join(save_path, f'fold{fold_num}', video[:video.index('_side')]+'.npy')
            if os.path.isfile(video_save_path):
                print(f'skipped {video}')
                continue
            print(f'inference for video {video}')
            features = []
            frames = sorted(os.listdir(os.path.join(frames_path, video)))
            # batch = []
            listdir = [os.path.join(frames_path, video, frame) for frame in frames]
            dataset = CustomImageDataset(listdir)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=12, pin_memory=True)
            # for i, frame in enumerate(frames):
            for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                # im = read_image(os.path.join(frames_path, video, frame))
                # im = transforms(im)
                # batch.append(im)
                # if len(batch) == 16 or i == (len(frames)-1):
                # b = torch.stack(batch).to(device)
                b, _ = data
                b = b.to(device)
                out = model(b).cpu().numpy()
                # out = model(im.unsqueeze(0)).squeeze(0).cpu().numpy()
                features.append(out)
                # batch = []
            features = np.concatenate(features).T
            # features = np.array(features).T
            np.save(video_save_path, features)
            print(features.shape)

