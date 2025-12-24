import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, img_size=256):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        self.image_dir = os.path.join(root_dir, split)
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size * 2)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # 分离标签图和真实图像
        width, height = image.size
        label = image.crop((0, 0, width // 2, height))
        photo = image.crop((width // 2, 0, width, height))
        
        # 应用变换
        if self.transform:
            label = self.transform(label)
            photo = self.transform(photo)
        
        return label, photo
    
    def split_image(self, image):
        """将拼接的图像分割为标签和真实图像"""
        width = image.shape[2]
        label = image[:, :, :width//2]
        photo = image[:, :, width//2:]
        return label, photo

def get_dataloaders(data_dir, batch_size=4, img_size=256):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size * 2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size * 2)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = CityscapesDataset(data_dir, split='train', transform=train_transform, img_size=img_size)
    val_dataset = CityscapesDataset(data_dir, split='val', transform=val_transform, img_size=img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader