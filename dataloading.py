import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import random
from scipy.ndimage import gaussian_filter, rotate

class HeatmapDataset(Dataset):
    def __init__(self, root_dir, transform=False, augment=False):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.transform = transform
        self.augment = augment
    
    def __len__(self):
        if self.augment:
            return len(self.files) * 4
        else:
            return len(self.files)

    def __getitem__(self, idx):
        original_idx = idx 
        augmentation_idx = idx

        if self.augment:
            original_idx = original_idx // 4
            augmentation_idx = augmentation_idx % 4

        file_path = os.path.join(self.root_dir, self.files[original_idx])
        data = np.load(file_path)
        filename = self.files[original_idx]

        img = data['imgs']
        keypoints = data['keypoints']

        img = torch.from_numpy(img).unsqueeze(0).float()
        
        keypoints = torch.from_numpy(keypoints).float()

        if self.transform:
            if random.random() > 0.4:
                img = torch.flip(img, [2])
                keypoints = torch.flip(keypoints, [2])
            
            if random.random() > 0.4:
                img = torch.flip(img, [1])
                keypoints = torch.flip(keypoints, [1])

            if random.random() >0.4:
                angle = random.uniform(-30, 30)
                img = torch.from_numpy(rotate(img.numpy(), angle, axes=(1, 2),reshape=False)).float()
                keypoints = torch.from_numpy(rotate(keypoints.numpy(), angle, axes=(1, 2), mode='nearest',reshape=False)).float()

        elif self.augment:
            if augmentation_idx == 1:
                # Flip horizontally
                img = torch.flip(img, [2])
                keypoints = torch.flip(keypoints, [2])
            elif augmentation_idx == 2:
                # Flip vertically
                img = torch.flip(img, [1])
                keypoints = torch.flip(keypoints, [1])
            elif augmentation_idx == 3:
                angle = random.uniform(-10, 10)
                img = torch.from_numpy(rotate(img.numpy(), angle, axes=(1, 2), reshape=False)).float()
                keypoints = torch.from_numpy(rotate(keypoints.numpy(), angle, axes=(1, 2), reshape=False)).float()

        return img, keypoints, filename

def get_loaders(batch_size, train=True):
    train_dataset = HeatmapDataset(root_dir='/home/shreya/scratch/HeatmapRegression/Regional_keypoints/Train_128_full_2d',transform=True)

    if train==True:    
        #TODO: needs to point to actual val dataset, also need to modify code to remove hardcoding paths
        test_dataset = HeatmapDataset(root_dir='/home/shreya/scratch/HeatmapRegression/Regional_keypoints/Train_128_full_2d')
    
    else:    
        test_dataset = HeatmapDataset(root_dir='/home/shreya/scratch/HeatmapRegression/Regional_keypoints/Test_128_full_2d')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

