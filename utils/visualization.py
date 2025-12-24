import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np
import os

def denormalize(tensor):
    """将归一化的图像反归一化"""
    return (tensor + 1) / 2

def save_triplet(label, generated, ground_truth, epoch, sample_idx, save_dir='results'):
    """保存三联图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 反归一化
    label = denormalize(label)
    generated = denormalize(generated)
    ground_truth = denormalize(ground_truth)
    
    # 转换为numpy
    label_np = label.cpu().numpy().transpose(1, 2, 0)
    generated_np = generated.cpu().numpy().transpose(1, 2, 0)
    ground_truth_np = ground_truth.cpu().numpy().transpose(1, 2, 0)
    
    # 创建三联图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(np.clip(label_np, 0, 1))
    axes[0].set_title('Label')
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(generated_np, 0, 1))
    axes[1].set_title(f'Generated (Epoch {epoch})')
    axes[1].axis('off')
    
    axes[2].imshow(np.clip(ground_truth_np, 0, 1))
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'triplet_epoch{epoch:03d}_sample{sample_idx:03d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_curves(losses, save_path='training_curves.png'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Generator loss
    axes[0, 0].plot(losses['G'], label='Generator Loss')
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator loss
    axes[0, 1].plot(losses['D'], label='Discriminator Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # GAN loss
    axes[1, 0].plot(losses['GAN'], label='GAN Loss')
    axes[1, 0].set_title('GAN Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Pixel loss
    axes[1, 1].plot(losses['pixel'], label='Pixel Loss')
    axes[1, 1].set_title('Pixel Loss (L1)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(metrics_dict, save_path='metrics_comparison.png'):
    """绘制不同模型的指标对比"""
    models = list(metrics_dict.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # PSNR比较
    psnr_values = [metrics_dict[m]['psnr'] for m in models]
    axes[0, 0].bar(models, psnr_values)
    axes[0, 0].set_title('PSNR Comparison (Higher is better)')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # SSIM比较
    ssim_values = [metrics_dict[m]['ssim'] for m in models]
    axes[0, 1].bar(models, ssim_values)
    axes[0, 1].set_title('SSIM Comparison (Higher is better)')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # MAE比较
    mae_values = [metrics_dict[m]['mae'] for m in models]
    axes[1, 0].bar(models, mae_values)
    axes[1, 0].set_title('MAE Comparison (Lower is better)')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # FID比较
    fid_values = [metrics_dict[m]['fid'] for m in models]
    axes[1, 1].bar(models, fid_values)
    axes[1, 1].set_title('FID Comparison (Lower is better)')
    axes[1, 1].set_ylabel('FID')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()