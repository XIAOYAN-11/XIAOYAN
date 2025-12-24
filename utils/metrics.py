import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import structural_similarity as ssim_skimage
from scipy import linalg
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d

def calculate_psnr(img1, img2):
    """计算PSNR"""
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    
    # 确保值在0-1范围内
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    psnr_value = psnr_skimage(img1, img2, data_range=1.0)
    return psnr_value

def calculate_ssim(img1, img2):
    """计算SSIM"""
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    if img1.shape[0] == 3:  # 如果是CHW格式
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
    
    ssim_value = ssim_skimage(img1, img2, channel_axis=-1 if img1.shape[-1] == 3 else None, data_range=1.0)
    return ssim_value

def calculate_mae(img1, img2):
    """计算MAE"""
    return torch.mean(torch.abs(img1 - img2)).item()

def calculate_fid(real_imgs, fake_imgs, device='cuda'):
    """计算FID分数"""
    # 使用Inception V3提取特征
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    
    # 提取特征
    real_features = []
    fake_features = []
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for img in real_imgs:
            img = transform(img).unsqueeze(0).to(device)
            feature = inception(img)
            real_features.append(feature.cpu().numpy())
        
        for img in fake_imgs:
            img = transform(img).unsqueeze(0).to(device)
            feature = inception(img)
            fake_features.append(feature.cpu().numpy())
    
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    
    # 计算均值和协方差
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # 计算FID
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def calculate_lpips(img1, img2, lpips_model):
    """计算LPIPS（感知损失）"""
    return lpips_model(img1, img2).mean().item()

def evaluate_model(model, dataloader, device='cuda', num_samples=10):
    """全面评估模型"""
    model.eval()
    
    psnr_values = []
    ssim_values = []
    mae_values = []
    all_real = []
    all_fake = []
    
    with torch.no_grad():
        for i, (labels, photos) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            labels = labels.to(device)
            photos = photos.to(device)
            
            fake_photos = model.generator(labels) if hasattr(model, 'generator') else model(labels)
            
            # 反归一化
            fake_photos = (fake_photos + 1) / 2
            photos = (photos + 1) / 2
            
            for j in range(len(labels)):
                psnr = calculate_psnr(fake_photos[j], photos[j])
                ssim = calculate_ssim(fake_photos[j], photos[j])
                mae = calculate_mae(fake_photos[j], photos[j])
                
                psnr_values.append(psnr)
                ssim_values.append(ssim)
                mae_values.append(mae)
                
                all_real.append(photos[j])
                all_fake.append(fake_photos[j])
    
    # 计算FID
    fid = calculate_fid(all_real, all_fake, device)
    
    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'mae': np.mean(mae_values),
        'fid': fid
    }