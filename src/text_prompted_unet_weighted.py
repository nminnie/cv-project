import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import clip

from data_preprocessing import PetDataset

# Set random seed for reproducibility
seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check GPU memory if available
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Available GPU memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")


def get_class_specific_text_features(clip_model, prompts_per_class, device):
    """
    Get text features for each class separately
    
    Args:
        clip_model: CLIP model
        prompts_per_class: List of lists of prompts, one list per class
        device: torch device
    
    Returns:
        Tensor of shape (num_classes, embed_dim)
    """
    all_text_features = []
    
    for class_prompts in prompts_per_class:
        # Tokenize all prompts for this class
        tokens = clip.tokenize(class_prompts).to(device)
        with torch.no_grad():
            # Get text features for all prompts
            class_features = clip_model.encode_text(tokens)
            # Normalize features
            class_features = class_features / class_features.norm(dim=1, keepdim=True)
            # Average the embeddings
            avg_features = class_features.mean(dim=0, keepdim=True)
            # Normalize again
            avg_features = avg_features / avg_features.norm(dim=1, keepdim=True)
        
        all_text_features.append(avg_features)
    
    # Concatenate all class features
    return torch.cat(all_text_features, dim=0)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class CLIPPromptedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features_start=32, text_embedding_dim=512):
        super(CLIPPromptedUNet, self).__init__()
        
        self.text_embedding_dim = text_embedding_dim
        
        # Encoder (Downsampling)
        self.encoder1 = DoubleConv(in_channels, features_start)  # 224 -> 224
        self.pool1 = nn.MaxPool2d(2)  # 224 -> 112
        self.encoder2 = DoubleConv(features_start, features_start*2)  # 112 -> 112
        self.pool2 = nn.MaxPool2d(2)  # 112 -> 56
        self.encoder3 = DoubleConv(features_start*2, features_start*4)  # 56 -> 56
        self.pool3 = nn.MaxPool2d(2)  # 56 -> 28
        self.encoder4 = DoubleConv(features_start*4, features_start*8)  # 28 -> 28
        self.pool4 = nn.MaxPool2d(2)  # 28 -> 14
        
        # Bottom - now includes text embedding integration
        self.bottom = DoubleConv(features_start*8, features_start*16)  # 14 -> 14
        
        # Text embedding projection layers
        self.text_proj = nn.Sequential(
            nn.Linear(text_embedding_dim, features_start*16),
            nn.ReLU(inplace=True)
        )
        
        # Text feature spatial integration
        self.text_spatial_proj = nn.Sequential(
            nn.Conv2d(features_start*16, features_start*16, kernel_size=1),
            nn.BatchNorm2d(features_start*16),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism
        self.query_proj = nn.Conv2d(features_start*16, features_start*16, kernel_size=1)
        self.key_proj = nn.Linear(features_start*16, features_start*16)
        self.value_proj = nn.Linear(features_start*16, features_start*16)
        self.attention_scale = 1.0 / (features_start*16)**0.5
        
        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(features_start*16, features_start*8, kernel_size=2, stride=2)  # 14 -> 28
        self.decoder4 = DoubleConv(features_start*16, features_start*8)  # 28 -> 28
        self.upconv3 = nn.ConvTranspose2d(features_start*8, features_start*4, kernel_size=2, stride=2)  # 28 -> 56
        self.decoder3 = DoubleConv(features_start*8, features_start*4)  # 56 -> 56
        self.upconv2 = nn.ConvTranspose2d(features_start*4, features_start*2, kernel_size=2, stride=2)  # 56 -> 112
        self.decoder2 = DoubleConv(features_start*4, features_start*2)  # 112 -> 112
        self.upconv1 = nn.ConvTranspose2d(features_start*2, features_start, kernel_size=2, stride=2)  # 112 -> 224
        self.decoder1 = DoubleConv(features_start*2, features_start)  # 224 -> 224
        
        # Final Convolution
        self.final_conv = nn.Conv2d(features_start, out_channels, kernel_size=1)  # 224 -> 224

    def apply_attention(self, image_features, text_features):
        """
        Apply cross-attention between image features and text features
        
        Args:
            image_features: Features from the image encoder (B, C, H, W)
            text_features: Projected text embeddings (B, C)
            
        Returns:
            Attention-enhanced features (B, C, H, W)
        """
        batch_size, channels, height, width = image_features.shape
        
        # Reshape image features for attention
        queries = self.query_proj(image_features)  # B, C, H, W
        queries = queries.flatten(2)  # B, C, H*W
        queries = queries.permute(0, 2, 1)  # B, H*W, C
        
        # Project text features
        keys = self.key_proj(text_features)  # B, C
        values = self.value_proj(text_features)  # B, C
        
        # Add dimensions for attention
        keys = keys.unsqueeze(1)  # B, 1, C
        values = values.unsqueeze(1)  # B, 1, C
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.attention_scale  # B, H*W, 1
        attention_weights = F.softmax(attention_scores, dim=-1)  # B, H*W, 1
        
        # Apply attention
        context = torch.matmul(attention_weights, values)  # B, H*W, C
        
        # Reshape back to spatial dimensions
        context = context.permute(0, 2, 1)  # B, C, H*W
        context = context.reshape(batch_size, channels, height, width)  # B, C, H, W
        
        # Add residual connection
        output = image_features + context
        
        return output

    def forward(self, x, text_embeddings):
        """
        Forward pass that incorporates CLIP text embeddings for guidance
        
        Args:
            x: Image tensor of shape (B, 3, H, W)
            text_embeddings: CLIP text embeddings of shape (B, num_classes, embed_dim)
                             where embed_dim is the CLIP embedding dimension
        """
        batch_size = x.shape[0]
        
        # Process text embeddings - reshape if necessary
        if text_embeddings.dim() == 3:
            # If we have embeddings for multiple classes, average them
            text_embeddings = text_embeddings.mean(dim=1)  # B, embed_dim
            
        # Project text embeddings to feature space
        text_features = self.text_proj(text_embeddings)  # B, features*16
        
        # Encoder
        enc1 = self.encoder1(x)  # 224
        enc2 = self.encoder2(self.pool1(enc1))  # 112
        enc3 = self.encoder3(self.pool2(enc2))  # 56
        enc4 = self.encoder4(self.pool3(enc3))  # 28
        
        # Bottom features
        bottom = self.bottom(self.pool4(enc4))  # 14 -> 14
        
        # Apply cross-attention between image features and text embeddings
        bottom = self.apply_attention(bottom, text_features)
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottom)  # 28
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)  # 56
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)  # 112
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)  # 224
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)


def get_class_specific_text_features(clip_model, prompts_per_class, device):
    """
    Get text features for each class separately
    
    Args:
        clip_model: CLIP model
        prompts_per_class: List of lists of prompts, one list per class
        device: torch device
    
    Returns:
        Tensor of shape (num_classes, embed_dim)
    """
    all_text_features = []
    
    for class_prompts in prompts_per_class:
        # Tokenize all prompts for this class
        tokens = clip.tokenize(class_prompts).to(device)
        with torch.no_grad():
            # Get text features for all prompts
            class_features = clip_model.encode_text(tokens)
            # Normalize features
            class_features = class_features / class_features.norm(dim=1, keepdim=True)
            # Average the embeddings
            avg_features = class_features.mean(dim=0, keepdim=True)
            # Normalize again
            avg_features = avg_features / avg_features.norm(dim=1, keepdim=True)
        
        all_text_features.append(avg_features)
    
    # Concatenate all class features
    return torch.cat(all_text_features, dim=0)  # (num_classes, embed_dim)


def train_text_prompted_model(model, train_loader, val_loader, text_features, run_path, num_epochs=20, cat_weight=2.1, device=device):
    """
    Train the text-prompted U-Net model.
    
    Args:
        model: text-prompted U-Net model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train
        cat_weight: Weight for the cat class
        device: Device to train on
        
    Returns:
        Trained model
    """

    # Add these lines at the beginning
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    accumulation_steps = 2
    
    print("Function started: train_text_prompted_unet")
    start_time = time.time()
    
    # Define class weights to address class imbalance [background, cat, dog]
    print("Setting up class weights...")
    class_weights = torch.tensor([1.0, cat_weight, 1.0], device=device)
    print(f"Class weights created: {time.time() - start_time:.2f}s")

    # Define loss function with class weights and ignore_index for white pixels
    print("Setting up criterion...")
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    print(f"Criterion created: {time.time() - start_time:.2f}s")
    
    # Define optimizer
    print("Setting up optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"Optimizer created: {time.time() - start_time:.2f}s")
    
    # Test data loading
    print("Testing data loading...")
    try:
        test_imgs, test_masks = next(iter(train_loader))
        print(f"Sample batch loaded - images: {test_imgs.shape}, masks: {test_masks.shape}")
        print(f"Data loading test: {time.time() - start_time:.2f}s")
        
        # Test batch to GPU transfer
        print("Testing batch GPU transfer...")
        test_imgs = test_imgs.to(device)
        test_masks = test_masks.to(device)
        print(f"GPU transfer test: {time.time() - start_time:.2f}s")
        
        # Test forward pass
        print("Testing forward pass...")
        with torch.no_grad():
            batch_size = test_imgs.shape[0]
            batch_text_features = text_features.unsqueeze(0).expand(batch_size, -1, -1)
            batch_text_features = batch_text_features.to(next(model.parameters()).dtype)
            test_output = model(test_imgs, batch_text_features)
        print(f"Forward pass test: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Error in data loading test: {e}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Track best model
    best_val_loss = float('inf')
    best_model = None
    
    # Training history
    train_losses = []
    val_losses = []
    train_iou = []
    val_iou = []
    train_dice = []
    val_dice = []
    prev_lr = optimizer.param_groups[0]['lr']
    
    # Create figures for live plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # GPU memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_allocated_start = torch.cuda.memory_allocated() / 1e9
        
        # Training phase
        model.train()
        running_loss = 0.0
        intersection = torch.zeros(3, device=device)
        union = torch.zeros(3, device=device)
        dice_intersection = torch.zeros(3, device=device)
        dice_sum = torch.zeros(3, device=device)
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=True)
        
        # Reset gradients at the start of each epoch
        optimizer.zero_grad()
        batch_count = 0
        
        for images, masks in train_pbar:
            batch_count += 1
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)

            # Expand text features for batch
            batch_size = images.shape[0]
            batch_text_features = text_features.unsqueeze(0).expand(batch_size, -1, -1)  # B, num_classes, embed_dim
            batch_text_features = batch_text_features.to(next(model.parameters()).dtype)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(images, batch_text_features)
                loss = criterion(outputs, masks)
            
            # Scale loss by accumulation steps and backward pass
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            
            # Update weights after accumulation_steps
            if batch_count % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Track loss (use the original loss value for tracking)
            running_loss += (loss.item() * accumulation_steps) * images.size(0)
            
            # Calculate metrics (use non-scaled outputs)
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                mask = (masks != 255)
                
                # Calculate IoU and Dice metrics
                for cls in range(3):
                    pred_cls = (predicted == cls) & mask
                    true_cls = (masks == cls) & mask
                    
                    intersection[cls] += (pred_cls & true_cls).sum().float()
                    union[cls] += (pred_cls | true_cls).sum().float()
                    
                    dice_intersection[cls] += (pred_cls & true_cls).sum().float()
                    dice_sum[cls] += pred_cls.sum().float() + true_cls.sum().float()
            
            # Update progress bar
            train_pbar.set_postfix(loss=loss.item() * accumulation_steps)
        
        # Make sure to update for any remaining batches (if dataset size % accumulation_steps != 0)
        if batch_count % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Calculate average training metrics
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Calculate IoU
        class_ious = []
        for cls in range(3):
            iou = intersection[cls] / union[cls] if union[cls] > 0 else 0
            class_ious.append(iou.item())
        mean_iou = sum(class_ious) / len(class_ious)
        train_iou.append(mean_iou)
        
        # Calculate Dice coefficient
        class_dice = []
        for cls in range(3):
            dice = (2 * dice_intersection[cls]) / dice_sum[cls] if dice_sum[cls] > 0 else 0
            class_dice.append(dice.item())
        mean_dice = sum(class_dice) / len(class_dice)
        train_dice.append(mean_dice)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_intersection = torch.zeros(3, device=device)
        val_union = torch.zeros(3, device=device)
        val_dice_intersection = torch.zeros(3, device=device)
        val_dice_sum = torch.zeros(3, device=device)
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False)
        
        with torch.no_grad():
            for images, masks in val_pbar:
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)

                # Expand text features for batch
                batch_size = images.shape[0]
                batch_text_features = text_features.unsqueeze(0).expand(batch_size, -1, -1)
                batch_text_features = batch_text_features.to(next(model.parameters()).dtype)

                # Forward pass
                outputs = model(images, batch_text_features)
                
                # Calculate loss
                loss = criterion(outputs, masks)
                
                # Track loss
                val_running_loss += loss.item() * images.size(0)
                
                # Calculate metrics
                _, predicted = torch.max(outputs, 1)
                mask = (masks != 255)
                
                # Calculate IoU
                for cls in range(3):  # 3 classes: background, cat, dog
                    pred_cls = (predicted == cls) & mask
                    true_cls = (masks == cls) & mask
                    
                    # Intersection and union
                    val_intersection[cls] += (pred_cls & true_cls).sum().float()
                    val_union[cls] += (pred_cls | true_cls).sum().float()
                    
                    # Dice coefficient
                    val_dice_intersection[cls] += (pred_cls & true_cls).sum().float()
                    val_dice_sum[cls] += pred_cls.sum().float() + true_cls.sum().float()
                
                # Update progress bar
                val_pbar.set_postfix(loss=loss.item())
        
        # Calculate average validation metrics
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate IoU
        val_class_ious = []
        for cls in range(3):
            iou = val_intersection[cls] / val_union[cls] if val_union[cls] > 0 else 0
            val_class_ious.append(iou.item())
        val_mean_iou = sum(val_class_ious) / len(val_class_ious)
        val_iou.append(val_mean_iou)
        
        # Calculate Dice coefficient
        val_class_dice = []
        for cls in range(3):
            dice = (2 * val_dice_intersection[cls]) / val_dice_sum[cls] if val_dice_sum[cls] > 0 else 0
            val_class_dice.append(dice.item())
        val_mean_dice = sum(val_class_dice) / len(val_class_dice)
        val_dice.append(val_mean_dice)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Check for learning rate changes
        current_lr = optimizer.param_groups[0]['lr']
        lr_changed = current_lr != prev_lr
        
        # GPU memory usage
        gpu_info = ""
        if torch.cuda.is_available():
            mem_allocated_peak = torch.cuda.max_memory_allocated() / 1e9
            mem_allocated_end = torch.cuda.memory_allocated() / 1e9
            gpu_info = f", Peak GPU Memory: {mem_allocated_peak:.2f} GB"
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.1f}s{gpu_info}")
        print(f"  Train: Loss={train_loss:.4f}, IoU={mean_iou:.4f}, Dice={mean_dice:.4f}")
        print(f"  Val: Loss={val_loss:.4f}, IoU={val_mean_iou:.4f}, Dice={val_mean_dice:.4f}")
        
        if lr_changed:
            print(f"  Learning rate changed from {prev_lr:.6f} to {current_lr:.6f}")
            prev_lr = current_lr
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            torch.save(model.state_dict(), f"{run_path}/text_prompted_unet_best.pth")
            print(f"  New best model saved with validation loss: {val_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), f'{run_path}/text_prompted_unet_final.pth')
    
    # Load best model
    model.load_state_dict(best_model)
    
    # Create final plots
    # Training and Validation Loss
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{run_path}/loss_curve.png')

    # IoU and Dice Scores
    plt.figure(figsize=(7, 5))
    plt.plot(train_iou, 'b--', label='Train IoU')
    plt.plot(val_iou, 'r--', label='Val IoU')
    plt.plot(train_dice, 'b-', label='Train Dice')
    plt.plot(val_dice, 'r-', label='Val Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('IoU and Dice Scores')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{run_path}/metrics_curve.png')
    
    return model, {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_iou': train_iou,
        'val_iou': val_iou,
        'train_dice': train_dice,
        'val_dice': val_dice
    }


def evaluate_model(model, dataloader, text_features, device):
    """
    Evaluate the text-prompted U-Net model.
    
    Args:
        model: Text-prompted U-Net model
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Initialize metrics
    correct = 0
    total = 0
    
    # Initialize IoU and Dice metrics - one for each class
    num_classes = 3  # background (0), cat (1), dog (2)
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    dice_intersection = torch.zeros(num_classes, device=device)
    dice_sum = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            # Expand text features for batch
            batch_size = images.shape[0]
            batch_text_features = text_features.unsqueeze(0).expand(batch_size, -1, -1)
            batch_text_features = batch_text_features.to(next(model.parameters()).dtype)
            
            # Get predictions
            outputs = model(images, batch_text_features)
            _, predicted = torch.max(outputs, 1)
            
            # Calculate accuracy (ignoring white pixels with value 255)
            mask = (masks != 255)
            correct += (predicted[mask] == masks[mask]).sum().item()
            total += mask.sum().item()
            
            # Calculate IoU and Dice for each class
            for cls in range(num_classes):
                pred_cls = (predicted == cls) & mask
                true_cls = (masks == cls) & mask
                
                # Intersection and union
                intersection[cls] += (pred_cls & true_cls).sum().float()
                union[cls] += (pred_cls | true_cls).sum().float()
                
                # Dice coefficient
                dice_intersection[cls] += (pred_cls & true_cls).sum().float()
                dice_sum[cls] += pred_cls.sum().float() + true_cls.sum().float()
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    # Calculate IoU for each class
    class_names = ["background", "cat", "dog"]
    class_ious = []
    class_dice = []
    
    for cls in range(num_classes):
        iou = intersection[cls] / union[cls] if union[cls] > 0 else 0
        class_ious.append(iou.item())
        
        dice = (2 * dice_intersection[cls]) / dice_sum[cls] if dice_sum[cls] > 0 else 0
        class_dice.append(dice.item())
        
        print(f"Class '{class_names[cls]}': IoU={iou:.4f}, Dice={dice:.4f}")
    
    # Calculate mean IoU and Dice
    mean_iou = sum(class_ious) / len(class_ious)
    mean_dice = sum(class_dice) / len(class_dice)
    
    print(f"Pixel Accuracy: {accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    
    return {
        "pixel_accuracy": accuracy,
        "class_ious": {class_names[i]: class_ious[i] for i in range(num_classes)},
        "mean_iou": mean_iou,
        "class_dice": {class_names[i]: class_dice[i] for i in range(num_classes)},
        "mean_dice": mean_dice
    }


if __name__ == "__main__":
    # Set paths and create datasets
    data_root = '../Dataset_augmented/'
    run_name = sys.argv[1]
    run_path = f'runs/text_prompted_unet/{run_name}'
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    # Create datasets - directly use train/val/test splits from the augmented dataset
    train_dataset = PetDataset(data_root, 'train')
    val_dataset = PetDataset(data_root, 'val')
    test_dataset = PetDataset(data_root, 'test')

    # Print dataset sizes
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    print(f"Checking dataset contents...")
    train_dir = os.path.join(data_root, 'train', 'color')
    print(f"Train image count: {len(os.listdir(train_dir))}")
    print(f"Checking first few images...")
    for i, f in enumerate(os.listdir(train_dir)[:5]):
        path = os.path.join(train_dir, f)
        size = os.path.getsize(path) / (1024*1024)  # Size in MB
        print(f"  {f}: {size:.2f} MB")

    # Create dataloaders
    batch_size = 64
    num_workers = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    
    # Load CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device)

    # Freeze CLIP parameters
    for param in clip_model.parameters():
        param.requires_grad = False

    # Define prompts per class
    background_prompts = [
        "background region with no pets",
        "the area surrounding",
        "scenery, furniture, floors, or walls"
    ]

    cat_prompts = [
        "a domestic cat with fur and whiskers",
        "a feline pet with pointed ears and a tail",
        "the complete shape and outline of a cat"
    ]

    dog_prompts = [
        "a domestic dog with fur and a snout",
        "a canine pet with distinctive ears",
        "the complete shape and outline of a dog"
    ]

    prompts_per_class = [background_prompts, cat_prompts, dog_prompts]

    text_features = get_class_specific_text_features(clip_model, prompts_per_class, device)

    # Create the model
    model = CLIPPromptedUNet(
        in_channels=3,
        out_channels=3,  # 3 classes: background, cat, dog
        features_start=32,
        text_embedding_dim=text_features.shape[1]  # CLIP text embedding dimension
    ).to(device)    

    # Train the model
    model, _ = train_text_prompted_model(
        model,
        train_loader,
        val_loader,
        text_features,
        run_path,
        num_epochs=20,
        cat_weight=2.1,
        device=device
    )

    print("\nEvaluating model on validation set:")
    val_results = evaluate_model(model, val_loader, text_features, device)

    print("\nEvaluating model on test set:")
    test_results = evaluate_model(model, test_loader, text_features, device)

    print("Training and evaluation completed and model saved!")