#!/usr/bin/env python3
"""
图像到图像生成实验 - 主脚本
使用方法：
1. 训练Pix2Pix模型：python main.py --mode train --model pix2pix --data_dir /path/to/data
2. 训练U-Net模型：python main.py --mode train --model unet --data_dir /path/to/data
3. 评估模型：python main.py --mode evaluate --data_dir /path/to/data
"""

import argparse
import os
import sys

def setup_environment():
    """设置环境"""
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='图像到图像生成实验')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'evaluate', 'test'],
                       help='运行模式')
    parser.add_argument('--model', type=str, default='pix2pix',
                       choices=['pix2pix', 'unet'],
                       help='模型类型')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据集路径')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='学习率')
    parser.add_argument('--checkpoint', type=str,
                       help='模型检查点路径')
    
    args = parser.parse_args()
    
    setup_environment()
    
    if args.mode == 'train':
        if args.model == 'pix2pix':
            from train import train_pix2pix
            
            # 训练参数
            train_args = argparse.Namespace(
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                img_size=256,
                checkpoint_dir='checkpoints',
                sample_dir='samples',
                val_interval=5,
                save_interval=10,
                val_samples=20
            )
            
            print('开始训练Pix2Pix模型...')
            train_pix2pix(train_args)
            
        elif args.model == 'unet':
            from train import train_unet
            
            train_args = argparse.Namespace(
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                img_size=256,
                checkpoint_dir='checkpoints',
                sample_dir='samples_unet',
                val_interval=10,
                save_interval=20,
                val_samples=20
            )
            
            print('开始训练U-Net模型...')
            train_unet(train_args)
    
    elif args.mode == 'evaluate':
        from evaluate import evaluate_all_models
        
        eval_args = argparse.Namespace(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            img_size=256,
            num_samples=50,
            models={
                'Pix2Pix': 'checkpoints/pix2pix_epoch100.pth',
                'U-Net': 'checkpoints/unet.pth'
            } if not args.checkpoint else {
                args.model: args.checkpoint
            }
        )
        
        print('开始评估模型...')
        evaluate_all_models(eval_args)
    
    elif args.mode == 'test':
        print('测试模式 - 生成样例图片')
        # 这里可以添加生成测试样例的代码
        pass

if __name__ == '__main__':
    main()