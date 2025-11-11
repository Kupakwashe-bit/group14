import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet50 for image classification')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='Number of classes in the dataset')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--fine-tune', action='store_true',
                        help='Fine-tune all layers')
    return parser.parse_args()

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image paths"""
    def __getitem__(self, index):
        path, _ = self.samples[index]
        img, target = super().__getitem__(index)
        return img, target, path

def get_dataloaders(data_dir, batch_size=32, val_split=0.1, num_workers=4):
    """Create train and validation dataloaders"""
    # Define data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = ImageFolderWithPaths(os.path.join(data_dir, 'train'))
    
    # Split into train and validation
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms to the datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset.dataset,  # Access the underlying dataset
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset.dataset,  # Access the underlying dataset
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.classes

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets, _) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            logger.info(f"Epoch: {epoch+1}, Batch: {batch_idx}/{len(dataloader)}, "
                       f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100. * correct / len(dataloader.dataset)
    
    if writer is not None:
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, epoch, writer=None):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100. * correct / len(dataloader.dataset)
    
    if writer is not None:
        writer.add_scalar('Loss/val', epoch_loss, epoch)
        writer.add_scalar('Accuracy/val', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Save model checkpoint"""
    torch.save(state, filename)

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup TensorBoard
    log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    # Get data loaders
    logger.info("Loading datasets...")
    train_loader, val_loader, classes = get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size
    )
    
    num_classes = len(classes)
    logger.info(f"Found {num_classes} classes: {classes}")
    
    # Save class names
    with open(os.path.join(args.output_dir, 'labels.txt'), 'w') as f:
        f.write('\n'.join(classes))
    
    # Initialize model
    logger.info("Initializing model...")
    model = models.resnet50(pretrained=args.pretrained)
    
    # Modify the final layer for the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Freeze all layers if fine-tuning
    if not args.fine_tune and args.pretrained:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the final layer
        for param in model.fc.parameters():
            param.requires_grad = True
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 60)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'optimizer': optimizer.state_dict(),
            'class_to_idx': train_loader.dataset.class_to_idx,
            'classes': classes
        }
        
        save_checkpoint(
            checkpoint, 
            os.path.join(args.output_dir, 'resnet50_ft.pt')
        )
        
        if is_best:
            save_checkpoint(
                checkpoint,
                os.path.join(args.output_dir, 'resnet50_best.pt')
            )
    
    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    writer.close()

if __name__ == "__main__":
    main()
