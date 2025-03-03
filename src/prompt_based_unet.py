import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from scipy.ndimage import gaussian_filter, map_coordinates

from data_preprocessing import JointTransform, PetDataset


class PointPromptedDataset(Dataset):
    def __init__(self, base_dataset, num_points=1, point_sampling='foreground_biased'):
        """
        Wrapper around the base dataset that adds point prompts.
        
        Args:
            base_dataset (Dataset): The base dataset providing images and masks
            num_points (int): Number of points to sample per image
            point_sampling (str): Strategy for sampling points:
                - 'random': Completely random points
                - 'foreground_biased': Higher probability of sampling foreground
                - 'class_balanced': Try to balance points across classes
        """
        self.base_dataset = base_dataset
        self.num_points = num_points
        self.point_sampling = point_sampling
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original image and mask
        image, mask = self.base_dataset[idx]
        
        # Sample points based on the selected strategy
        points = self._sample_points(mask)
        
        # Create a heatmap from the points
        heatmap = self._create_heatmap(points, mask.shape)
        
        return image, heatmap, mask
    
    def _sample_points(self, mask):
        """Sample points based on the selected strategy"""
        h, w = mask.shape
        points = []
        
        for _ in range(self.num_points):
            if self.point_sampling == 'random':
                # Completely random point
                y = random.randint(0, h-1)
                x = random.randint(0, w-1)
                
            elif self.point_sampling == 'foreground_biased':
                # 80% chance to sample from foreground (non-background)
                if random.random() < 0.8:
                    # Get non-background/non-ignore indices
                    valid_mask = (mask != 0) & (mask != 255)
                    if valid_mask.sum() > 0:
                        # Convert to indices
                        valid_indices = torch.nonzero(valid_mask)
                        # Sample a random index
                        random_idx = random.randint(0, valid_indices.shape[0]-1)
                        y, x = valid_indices[random_idx]
                    else:
                        # Fallback to random if no valid foreground
                        y = random.randint(0, h-1)
                        x = random.randint(0, w-1)
                else:
                    # Sample from anywhere
                    y = random.randint(0, h-1)
                    x = random.randint(0, w-1)
                    
            elif self.point_sampling == 'class_balanced':
                # Try to balance points across classes
                # Get class distribution
                class_counts = {}
                for c in [0, 1, 2]:  # Background, cat, dog
                    class_counts[c] = (mask == c).sum().item()
                
                # Calculate sampling probabilities (inverse to class frequency)
                total_pixels = sum(class_counts.values())
                probs = {}
                for c in class_counts:
                    if class_counts[c] > 0:
                        probs[c] = 1.0 - (class_counts[c] / total_pixels)
                    else:
                        probs[c] = 0.0
                
                # Normalize probabilities
                prob_sum = sum(probs.values())
                if prob_sum > 0:
                    for c in probs:
                        probs[c] /= prob_sum
                else:
                    # Fallback to equal probabilities
                    for c in probs:
                        probs[c] = 1.0 / len(probs)
                
                # Sample a class based on probabilities
                classes = list(probs.keys())
                class_probs = [probs[c] for c in classes]
                target_class = random.choices(classes, weights=class_probs, k=1)[0]
                
                # Sample a point from the selected class
                class_mask = (mask == target_class)
                if class_mask.sum() > 0:
                    indices = torch.nonzero(class_mask)
                    random_idx = random.randint(0, indices.shape[0]-1)
                    y, x = indices[random_idx]
                else:
                    # Fallback to random
                    y = random.randint(0, h-1)
                    x = random.randint(0, w-1)
            
            points.append((y.item() if isinstance(y, torch.Tensor) else y, 
                           x.item() if isinstance(x, torch.Tensor) else x))
        
        return points
    
    def _create_heatmap(self, points, shape):
        """Create a Gaussian heatmap from points"""
        heatmap = torch.zeros(shape)
        
        # Gaussian parameters
        sigma = 3.0
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(0, shape[0]), 
            torch.arange(0, shape[1]),
            indexing='ij'
        )
        
        # Add each point's Gaussian
        for y, x in points:
            # Calculate squared distance
            squared_dist = (y_grid - y) ** 2 + (x_grid - x) ** 2
            
            # Apply Gaussian function
            gaussian = torch.exp(-squared_dist / (2 * sigma ** 2))
            
            # Add to heatmap
            heatmap = torch.maximum(heatmap, gaussian)
        
        return heatmap.unsqueeze(0)  # Add channel dimension (C=1)


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

class PointPromptUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features_start=32):
        super(PointPromptUNet, self).__init__()
        
        # Encoder for the point heatmap (single channel)
        self.point_encoder = nn.Sequential(
            nn.Conv2d(1, features_start // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features_start // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features_start // 2, features_start // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features_start // 2),
            nn.ReLU(inplace=True)
        )
        
        # Starting with 128x128 input
        # The image encoder now takes 3 channels (RGB) + features_start//2 (point features)
        combined_channels = in_channels + features_start // 2
        
        # Encoder (Downsampling)
        self.encoder1 = DoubleConv(combined_channels, features_start)  # 128 -> 128
        self.pool1 = nn.MaxPool2d(2)  # 128 -> 64
        self.encoder2 = DoubleConv(features_start, features_start*2)  # 64 -> 64
        self.pool2 = nn.MaxPool2d(2)  # 64 -> 32
        self.encoder3 = DoubleConv(features_start*2, features_start*4)  # 32 -> 32
        self.pool3 = nn.MaxPool2d(2)  # 32 -> 16
        self.encoder4 = DoubleConv(features_start*4, features_start*8)  # 16 -> 16
        self.pool4 = nn.MaxPool2d(2)  # 16 -> 8
        
        # Bottom
        self.bottom = DoubleConv(features_start*8, features_start*16)  # 8 -> 8
        
        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(features_start*16, features_start*8, kernel_size=2, stride=2)  # 8 -> 16
        self.decoder4 = DoubleConv(features_start*16, features_start*8)  # 16 -> 16
        self.upconv3 = nn.ConvTranspose2d(features_start*8, features_start*4, kernel_size=2, stride=2)  # 16 -> 32
        self.decoder3 = DoubleConv(features_start*8, features_start*4)  # 32 -> 32
        self.upconv2 = nn.ConvTranspose2d(features_start*4, features_start*2, kernel_size=2, stride=2)  # 32 -> 64
        self.decoder2 = DoubleConv(features_start*4, features_start*2)  # 64 -> 64
        self.upconv1 = nn.ConvTranspose2d(features_start*2, features_start, kernel_size=2, stride=2)  # 64 -> 128
        self.decoder1 = DoubleConv(features_start*2, features_start)  # 128 -> 128
        
        # Final Convolution
        self.final_conv = nn.Conv2d(features_start, out_channels, kernel_size=1)  # 128 -> 128

    def forward(self, x, point_heatmap):
        # First encode the point heatmap
        point_features = self.point_encoder(point_heatmap)
        
        # Concatenate image and point features
        x_combined = torch.cat([x, point_features], dim=1)
        
        # Encoder
        enc1 = self.encoder1(x_combined)  # 128
        enc2 = self.encoder2(self.pool1(enc1))  # 64
        enc3 = self.encoder3(self.pool2(enc2))  # 32
        enc4 = self.encoder4(self.pool3(enc3))  # 16
        
        # Bottom
        bottom = self.bottom(self.pool4(enc4))  # 8
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottom)  # 16
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)  # 32
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)  # 64
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)  # 128
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)


def train_model(
    model, 
    train_dataset, 
    val_dataset, 
    batch_size=8, 
    num_epochs=50, 
    learning_rate=1e-4, 
    device='cuda',
    save_path='./checkpoints',
    log_dir='./logs'
):
    """
    Train the point-prompted segmentation model
    """
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Setup loss function (ignore index 255)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # Setup optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Track best validation loss
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    
    # Move model to device
    model = model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct_pixels = 0
        train_total_pixels = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
        for images, heatmaps, masks in train_loop:
            # Move data to device
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, heatmaps)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backpropagation
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * images.size(0)
            
            # Calculate accuracy (ignore 255)
            preds = torch.argmax(outputs, dim=1)
            valid_mask = masks != 255
            train_correct_pixels += (preds[valid_mask] == masks[valid_mask]).sum().item()
            train_total_pixels += valid_mask.sum().item()
            
            # Update progress bar
            train_loop.set_postfix(loss=loss.item(), 
                                  acc=train_correct_pixels/max(1, train_total_pixels))
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracy = train_correct_pixels / max(1, train_total_pixels)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct_pixels = 0
        val_total_pixels = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)")
            for images, heatmaps, masks in val_loop:
                # Move data to device
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images, heatmaps)
                
                # Calculate loss
                loss = criterion(outputs, masks)
                
                # Track metrics
                val_loss += loss.item() * images.size(0)
                
                # Calculate accuracy (ignore 255)
                preds = torch.argmax(outputs, dim=1)
                valid_mask = masks != 255
                val_correct_pixels += (preds[valid_mask] == masks[valid_mask]).sum().item()
                val_total_pixels += valid_mask.sum().item()
                
                # Update progress bar
                val_loop.set_postfix(loss=loss.item(), 
                                   acc=val_correct_pixels/max(1, val_total_pixels))
        
        # Calculate epoch metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracy = val_correct_pixels / max(1, val_total_pixels)

        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        torch.save(checkpoint, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(save_path, 'best_model.pth'))
            
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    return model


if __name__ == "__main__":
    # Define image sizes
    img_size = 128
    crop_size = 112

    # Create transforms
    joint_transform_train = JointTransform(img_size=img_size, crop_size=crop_size)

    # Simple transform for validation/test (no augmentation)
    def val_transform(image, mask):
        # Resize both to same size
        image = TF.resize(image, (img_size, img_size))
        mask = TF.resize(mask, (img_size, img_size), interpolation=TF.InterpolationMode.NEAREST)
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

    # Create datasets
    data_root = '../Dataset/'  # Adjust path as needed
    trainval_dataset = PetDataset(data_root, 'TrainVal', transform=joint_transform_train)
    train_size = int(0.8 * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size

    # For training/validation split
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(trainval_dataset)), [train_size, val_size]
    )

    # Create datasets with appropriate transforms
    train_dataset = PetDataset(data_root, 'TrainVal', transform=joint_transform_train)
    train_dataset.image_files = [trainval_dataset.image_files[i] for i in train_indices.indices]

    val_dataset = PetDataset(data_root, 'TrainVal', transform=val_transform)
    val_dataset.image_files = [trainval_dataset.image_files[i] for i in val_indices.indices]

    test_dataset = PetDataset(data_root, 'Test', transform=val_transform)

    # Create point-prompted datasets
    point_train_dataset = PointPromptedDataset(train_dataset, num_points=1, point_sampling='foreground_biased')
    point_val_dataset = PointPromptedDataset(val_dataset, num_points=1, point_sampling='foreground_biased')
    point_test_dataset = PointPromptedDataset(test_dataset, num_points=1, point_sampling='foreground_biased')

    print("Training set size:", len(point_train_dataset))
    print("Validation set size:", len(point_val_dataset))
    print("Test set size:", len(point_test_dataset))

    # Create model (3 output channels for background, cat, dog)
    model = PointPromptUNet(in_channels=3, out_channels=3, features_start=32)

    # Train model
    trained_model = train_model(
        model=model,
        train_dataset=point_train_dataset,
        val_dataset=point_val_dataset,
        batch_size=8,
        num_epochs=20,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_path='./checkpoints',
        log_dir='./logs'
    )

    # Save final model
    torch.save(trained_model.state_dict(), 'point_prompt_segmentation_model.pth')
    print("Training completed.")
