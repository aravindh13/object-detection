"""
Training script for two-stream RGB + thermal object detection.
Supports mixed precision training, cosine LR scheduling, and checkpointing.
"""

import os
import argparse
import time
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from models.two_stream_yolo import create_model
from datasets.rgbt_dataset import RGBTThermalDataset, collate_fn
from losses.yolo_loss import YOLOLoss


class CosineScheduler:
    """Cosine learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def save_checkpoint(model: nn.Module, 
                   optimizer: optim.Optimizer, 
                   scheduler: CosineScheduler,
                   epoch: int, 
                   loss: float, 
                   save_path: str,
                   is_best: bool = False):
    """Save model checkpoint."""
    # Save only essential components to avoid custom class issues
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    # Save regular checkpoint
    torch.save(checkpoint, save_path)
    
    # Save best model (only model weights for inference)
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
        # Save only model weights for inference
        torch.save(model.state_dict(), best_path)
        print(f"New best model saved at epoch {epoch}")


def load_checkpoint(model: nn.Module, 
                   optimizer: optim.Optimizer, 
                   checkpoint_path: str) -> int:
    """Load model checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch
    return 0


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                loss_fn: YOLOLoss,
                scaler: GradScaler,
                device: torch.device,
                epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_obj_loss = 0
    total_cls_loss = 0
    total_box_loss = 0
    total_temp_loss = 0
    
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        rgb_images = batch['rgb_images'].to(device)
        thermal_images = batch['thermal_images'].to(device)
        boxes = batch['boxes'].to(device)
        class_ids = batch['class_ids'].to(device)
        
        # Prepare targets
        targets = {
            'boxes': boxes,
            'class_ids': class_ids,
            'image_ids': batch['image_ids']
        }
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            predictions = model(rgb_images, thermal_images)
            loss_dict = loss_fn(predictions, targets)
        
        # Backward pass
        scaler.scale(loss_dict['total_loss']).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        total_loss += loss_dict['total_loss'].item()
        total_obj_loss += loss_dict['obj_loss'].item()
        total_cls_loss += loss_dict['cls_loss'].item()
        total_box_loss += loss_dict['box_loss'].item()
        total_temp_loss += loss_dict['temp_loss'].item()
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                  f'Loss: {loss_dict["total_loss"].item():.4f}, '
                  f'Obj: {loss_dict["obj_loss"].item():.4f}, '
                  f'Cls: {loss_dict["cls_loss"].item():.4f}, '
                  f'Box: {loss_dict["box_loss"].item():.4f}, '
                  f'Temp: {loss_dict["temp_loss"].item():.4f}')
    
    # Average losses
    avg_losses = {
        'total_loss': total_loss / num_batches,
        'obj_loss': total_obj_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'box_loss': total_box_loss / num_batches,
        'temp_loss': total_temp_loss / num_batches
    }
    
    return avg_losses


def validate_epoch(model: nn.Module,
                   dataloader: DataLoader,
                   loss_fn: YOLOLoss,
                   device: torch.device) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0
    total_obj_loss = 0
    total_cls_loss = 0
    total_box_loss = 0
    total_temp_loss = 0
    
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            rgb_images = batch['rgb_images'].to(device)
            thermal_images = batch['thermal_images'].to(device)
            boxes = batch['boxes'].to(device)
            class_ids = batch['class_ids'].to(device)
            
            # Prepare targets
            targets = {
                'boxes': boxes,
                'class_ids': class_ids,
                'image_ids': batch['image_ids']
            }
            
            # Forward pass
            predictions = model(rgb_images, thermal_images)
            loss_dict = loss_fn(predictions, targets)
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            total_obj_loss += loss_dict['obj_loss'].item()
            total_cls_loss += loss_dict['cls_loss'].item()
            total_box_loss += loss_dict['box_loss'].item()
            total_temp_loss += loss_dict['temp_loss'].item()
    
    # Average losses
    avg_losses = {
        'total_loss': total_loss / num_batches,
        'obj_loss': total_obj_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'box_loss': total_box_loss / num_batches,
        'temp_loss': total_temp_loss / num_batches
    }
    
    return avg_losses


def main():
    parser = argparse.ArgumentParser(description='Train Two-Stream RGB+Thermal Object Detector')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--save_dir', type=str, default='runs', help='Directory to save checkpoints')
    parser.add_argument('--exp_name', type=str, default='exp', help='Experiment name')
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override config with command line arguments
    config.update({
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'warmup_epochs': args.warmup_epochs,
        'num_classes': config.get('num_classes', 80),
        'input_size': config.get('input_size', 640),
        'label_format': config.get('label_format', 'yolo')
    })
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_dir = Path(args.save_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    writer = SummaryWriter(exp_dir / 'tensorboard')
    
    # Create datasets
    print(f"Creating dataset from: {args.data_dir}")
    train_dataset = RGBTThermalDataset(
        data_dir=args.data_dir,
        label_format=config['label_format'],
        input_size=config['input_size'],
        augment=True
    )
    
    print(f"Dataset created successfully!")
    print(f"Total dataset size: {len(train_dataset)}")
    
    if len(train_dataset) == 0:
        print("ERROR: Dataset is empty!")
        print("Please check:")
        print("1. Data directory exists:", os.path.exists(args.data_dir))
        print("2. RGB directory exists:", os.path.exists(os.path.join(args.data_dir, "rgb")))
        print("3. Thermal directory exists:", os.path.exists(os.path.join(args.data_dir, "thermal")))
        print("4. Labels directory exists:", os.path.exists(os.path.join(args.data_dir, "labels")))
        if os.path.exists(os.path.join(args.data_dir, "rgb")):
            rgb_files = [f for f in os.listdir(os.path.join(args.data_dir, "rgb")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"5. RGB files found: {len(rgb_files)}")
        if os.path.exists(os.path.join(args.data_dir, "thermal")):
            thermal_files = [f for f in os.listdir(os.path.join(args.data_dir, "thermal")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"6. Thermal files found: {len(thermal_files)}")
        if os.path.exists(os.path.join(args.data_dir, "labels")):
            label_files = [f for f in os.listdir(os.path.join(args.data_dir, "labels")) if f.lower().endswith(('.txt', '.json'))]
            print(f"7. Label files found: {len(label_files)}")
        return
    
    val_dataset = RGBTThermalDataset(
        data_dir=args.data_dir,
        label_format=config['label_format'],
        input_size=config['input_size'],
        augment=False
    )
    
    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = create_model(
        num_classes=config['num_classes'],
        input_size=config['input_size'],
        pretrained=True
    )
    model = model.to(device)
    
    print(f"Model created with {model.get_model_size()['total_parameters']:,} parameters")
    
    # Create loss function
    loss_fn = YOLOLoss(
        num_classes=config['num_classes'],
        num_anchors=3,
        input_size=config['input_size']
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=1e-4
    )
    
    # Create scheduler
    scheduler = CosineScheduler(
        optimizer=optimizer,
        warmup_epochs=config['warmup_epochs'],
        total_epochs=config['epochs'],
        base_lr=config['lr']
    )
    
    # Create scaler for mixed precision
    scaler = GradScaler()
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume)
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch}/{config['epochs']-1}")
        print("-" * 50)
        
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, epoch
        )
        
        # Validate
        val_losses = validate_epoch(model, val_loader, loss_fn, device)
        
        # Log to TensorBoard
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        writer.add_scalar('Train/Total_Loss', train_losses['total_loss'], epoch)
        writer.add_scalar('Train/Obj_Loss', train_losses['obj_loss'], epoch)
        writer.add_scalar('Train/Cls_Loss', train_losses['cls_loss'], epoch)
        writer.add_scalar('Train/Box_Loss', train_losses['box_loss'], epoch)
        writer.add_scalar('Train/Temp_Loss', train_losses['temp_loss'], epoch)
        
        writer.add_scalar('Val/Total_Loss', val_losses['total_loss'], epoch)
        writer.add_scalar('Val/Obj_Loss', val_losses['obj_loss'], epoch)
        writer.add_scalar('Val/Cls_Loss', val_losses['cls_loss'], epoch)
        writer.add_scalar('Val/Box_Loss', val_losses['box_loss'], epoch)
        writer.add_scalar('Val/Temp_Loss', val_losses['temp_loss'], epoch)
        
        # Print epoch summary
        print(f"Train Loss: {train_losses['total_loss']:.4f}")
        print(f"Val Loss: {val_losses['total_loss']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_losses['total_loss'] < best_loss
        if is_best:
            best_loss = val_losses['total_loss']
        
        checkpoint_path = exp_dir / f'checkpoint_epoch_{epoch}.pth'
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_losses['total_loss'],
            str(checkpoint_path), is_best
        )
        
        # Save config
        with open(exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours")
    print(f"Best validation loss: {best_loss:.4f}")
    
    writer.close()


if __name__ == "__main__":
    main()

