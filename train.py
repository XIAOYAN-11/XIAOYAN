import os
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.dataloader import get_dataloaders
from models.pix2pix import Pix2Pix
from models.unet_simple import UNetSimple
from utils.visualization import save_triplet, plot_training_curves
from utils.metrics import evaluate_model

def train_pix2pix(args):
    """训练Pix2Pix模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # 加载数据
    train_loader, val_loader = get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        img_size=args.img_size
    )
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    
    # 初始化模型
    model = Pix2Pix(
        in_channels=3, 
        out_channels=3, 
        lr=args.lr, 
        device=device
    )
    
    # 训练记录
    losses = {
        'G': [], 'D': [], 'GAN': [], 'pixel': []
    }
    val_metrics = []
    
    # 训练循环
    for epoch in range(args.epochs):
        model.generator.train()
        model.discriminator.train()
        
        epoch_losses = {'G': 0, 'D': 0, 'GAN': 0, 'pixel': 0}
        
        # 训练阶段
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for i, (labels, photos) in enumerate(pbar):
            # 训练一步
            loss_dict = model.train_step(labels, photos)
            
            # 累积损失
            for key in epoch_losses:
                # epoch_losses[key] += loss_dict[key]
                epoch_losses[key] += loss_dict[f"loss_{key}"]
            
            # 更新进度条
            pbar.set_postfix({
                'loss_G': f"{loss_dict['loss_G']:.4f}",
                'loss_D': f"{loss_dict['loss_D']:.4f}"
            })
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
            losses[key].append(epoch_losses[key])
        
        # 验证阶段
        if (epoch + 1) % args.val_interval == 0:
            print(f'\nValidating epoch {epoch+1}...')
            metrics = evaluate_model(
                model, val_loader, device=device, num_samples=args.val_samples
            )
            val_metrics.append(metrics)
            
            print(f'Validation Metrics - PSNR: {metrics["psnr"]:.2f}, '
                  f'SSIM: {metrics["ssim"]:.4f}, MAE: {metrics["mae"]:.4f}, '
                  f'FID: {metrics["fid"]:.2f}')
            
            # 保存样例
            model.generator.eval()
            with torch.no_grad():
                val_labels, val_photos = next(iter(val_loader))
                val_labels = val_labels.to(device)
                val_photos = val_photos.to(device)
                
                fake_photos = model.generator(val_labels)
                
                for i in range(min(3, len(val_labels))):
                    save_triplet(
                        val_labels[i], 
                        fake_photos[i], 
                        val_photos[i],
                        epoch + 1, 
                        i,
                        save_dir=args.sample_dir
                    )
        
        # 保存模型
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': model.generator.state_dict(),
                'discriminator_state_dict': model.discriminator.state_dict(),
                'optimizer_G_state_dict': model.optimizer_G.state_dict(),
                'optimizer_D_state_dict': model.optimizer_D.state_dict(),
                'losses': losses,
                'val_metrics': val_metrics
            }
            torch.save(
                checkpoint, 
                os.path.join(args.checkpoint_dir, f'pix2pix_epoch{epoch+1}.pth')
            )
            print(f'Model saved at epoch {epoch+1}')
    
    # 绘制训练曲线
    plot_training_curves(losses, save_path=os.path.join(args.sample_dir, 'training_curves.png'))
    
    # 最终评估
    print('\nFinal evaluation...')
    final_metrics = evaluate_model(
        model, val_loader, device=device, num_samples=50
    )
    print(f'Final Metrics: {final_metrics}')
    
    return model, losses, val_metrics

def train_unet(args):
    """训练简单的U-Net模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_loader, val_loader = get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        img_size=args.img_size
    )
    
    # 初始化模型
    model = UNetSimple(in_channels=3, out_channels=3).to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    losses = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for labels, photos in pbar:
            labels, photos = labels.to(device), photos.to(device)
            
            optimizer.zero_grad()
            outputs = model(labels)
            loss = criterion(outputs, photos)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        losses.append(epoch_loss / len(train_loader))
        print(f'Epoch {epoch+1}, Loss: {losses[-1]:.4f}')
        
        # 保存样例
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_labels, val_photos = next(iter(val_loader))
                val_labels = val_labels.to(device)
                outputs = model(val_labels)
                
                for i in range(min(3, len(val_labels))):
                    save_triplet(
                        val_labels[i], 
                        outputs[i], 
                        val_photos[i],
                        epoch + 1, 
                        i,
                        save_dir=f'{args.sample_dir}_unet'
                    )
    
    return model, losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pix2pix', choices=['pix2pix', 'unet'])
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Cityscapes dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--val_samples', type=int, default=20)
    
    args = parser.parse_args()
    
    if args.model == 'pix2pix':
        train_pix2pix(args)
    elif args.model == 'unet':
        train_unet(args)

if __name__ == '__main__':
    main()