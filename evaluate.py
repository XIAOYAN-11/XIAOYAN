import os
import torch
import argparse
import json
from models.pix2pix import PixPix, GeneratorUNet
from models.unet_simple import UNetSimple
from utils.dataloader import get_dataloaders
from utils.metrics import evaluate_model
from utils.visualization import plot_metrics_comparison

def load_model(model_type, checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    if model_type == 'pix2pix':
        model = PixPix(device=device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        return model
    elif model_type == 'unet':
        model = UNetSimple().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        return model
    elif model_type == 'pix2pix_generator':
        generator = GeneratorUNet().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        return generator
    else:
        raise ValueError(f'Unknown model type: {model_type}')

def evaluate_all_models(args):
    """评估所有模型并进行比较"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载验证数据
    _, val_loader = get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        img_size=args.img_size
    )
    
    metrics_results = {}
    
    # 评估每个模型
    for model_name, checkpoint_path in args.models.items():
        print(f'\nEvaluating {model_name}...')
        
        # 加载模型
        if 'pix2pix' in model_name.lower():
            model = load_model('pix2pix', checkpoint_path, device)
        elif 'unet' in model_name.lower():
            model = load_model('unet', checkpoint_path, device)
        else:
            print(f'Unknown model type for {model_name}, skipping...')
            continue
        
        # 评估
        metrics = evaluate_model(
            model, val_loader, device=device, num_samples=args.num_samples
        )
        
        metrics_results[model_name] = metrics
        
        print(f'{model_name} - PSNR: {metrics["psnr"]:.2f}, '
              f'SSIM: {metrics["ssim"]:.4f}, MAE: {metrics["mae"]:.4f}, '
              f'FID: {metrics["fid"]:.2f}')
    
    # 保存结果
    with open('evaluation_results.json', 'w') as f:
        json.dump(metrics_results, f, indent=2)
    
    # 绘制比较图
    plot_metrics_comparison(metrics_results, save_path='metrics_comparison.png')
    
    print('\nEvaluation completed. Results saved to evaluation_results.json')
    
    return metrics_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples for evaluation')
    parser.add_argument('--models', type=str, nargs='+', 
                       help='List of model:checkpoint pairs, e.g., pix2pix:checkpoints/pix2pix.pth unet:checkpoints/unet.pth')
    
    args = parser.parse_args()
    
    # 解析模型参数
    if args.models:
        model_dict = {}
        for item in args.models:
            name, path = item.split(':')
            model_dict[name] = path
        args.models = model_dict
    else:
        # 默认模型
        args.models = {
            'Pix2Pix': 'checkpoints/pix2pix_epoch100.pth',
            'U-Net': 'checkpoints/unet.pth'
        }
    
    evaluate_all_models(args)

if __name__ == '__main__':
    main()