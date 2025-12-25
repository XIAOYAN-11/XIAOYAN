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
    """全面评估模型 - 已修复Pix2Pix的eval()调用错误"""

    # 关键修复：安全地设置模型为评估模式
    if hasattr(model, 'generator'):
        # 对于Pix2Pix类对象，只设置其生成器为评估模式
        model.generator.eval()
        use_generator = True
        print("检测到Pix2Pix类，使用generator进行评估")
    elif hasattr(model, 'eval'):
        # 对于普通PyTorch模型
        model.eval()
        use_generator = False
        print("检测到普通模型，使用model.eval()")
    else:
        # 如果模型既不是Pix2Pix也没有eval方法，直接使用
        use_generator = False
        print("警告：模型没有标准的eval方法，直接进行评估")

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

            # 关键修复：根据模型类型选择正确的调用方式
            if use_generator:
                fake_photos = model.generator(labels)
            else:
                fake_photos = model(labels)

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
    fid = calculate_fid(all_real, all_fake, device) if all_real else 999.0

    # 关键修复：安全地恢复模型为训练模式
    if hasattr(model, 'generator'):
        model.generator.train()
    elif hasattr(model, 'train'):
        model.train()

    return {
        'psnr': np.mean(psnr_values) if psnr_values else 0.0,
        'ssim': np.mean(ssim_values) if ssim_values else 0.0,
        'mae': np.mean(mae_values) if mae_values else 0.0,
        'fid': fid
    }


# 添加测试代码
if __name__ == '__main__':
    print("=" * 50)
    print("metrics.py 已修复Pix2Pix兼容性问题")
    print("主要修复内容:")
    print("1. 检测Pix2Pix类并正确使用generator进行评估")
    print("2. 避免对Pix2Pix对象调用不存在的eval()方法")
    print("3. 正确恢复模型训练模式")
    print("=" * 50)