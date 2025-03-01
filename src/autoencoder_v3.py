import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from data_preprocessing import JointTransform, PetDataset

# Set random seed for reproducibility
random.seed(100)
np.random.seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed(100)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



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

# Print dataset sizes
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Create dataloaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Define the Encoder part of the autoencoder
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # First convolution block
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64
            
            # Second convolution block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32
            
            # Third convolution block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            
            # Fourth convolution block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
        )
        
        # Feature dimension after encoding
        self.feature_dim = 512 * 8 * 8  # For 112x112 input images
        
        # Bottleneck fully connected layer
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, latent_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Get features from convolutional layers
        features = self.encoder(x)
        # Pass through bottleneck to get latent representation
        latent = self.bottleneck(features)
        return features, latent

# Define the Decoder for reconstruction
class ReconstructionDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super(ReconstructionDecoder, self).__init__()
        
        # Feature dimension after encoding
        self.feature_dim = 512 * 8 * 8  # For 112x112 input
        
        # Expand from latent space back to feature map dimensions
        self.expand = nn.Sequential(
            nn.Linear(latent_dim, self.feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Reshape dimensions for deconvolution
        self.reshape = lambda x: x.view(-1, 512, 8, 8)
        
        # Deconvolution / upsampling layers
        self.decoder = nn.Sequential(
            # First upsampling block
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # 14x14
            
            # Second upsampling block
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # 28x28
            
            # Third upsampling block
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # 56x56
            
            # Final upsampling block to original image size
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 112x112, normalize to [0,1]
        )
        
    def forward(self, features, latent):
        # Expand latent representation
        expanded = self.expand(latent)
        # Reshape to proper dimensions for deconvolution
        reshaped = self.reshape(expanded)
        # Decode back to image
        output = self.decoder(reshaped)
        return output

# Define the Autoencoder by combining Encoder and Decoder
class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = ReconstructionDecoder(latent_dim, in_channels)
        
    def forward(self, x):
        features, latent = self.encoder(x)
        reconstructed = self.decoder(features, latent)
        return reconstructed, features, latent

# Define the Segmentation Decoder
class SegmentationDecoder(nn.Module):
    def __init__(self, num_classes=3):
        super(SegmentationDecoder, self).__init__()
        
        # Use transposed convolutions to upsample feature maps
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        
        # Final layer to produce segmentation map
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, features):
        # Upsample 4 times to match original image size
        x = F.relu(self.bn1(self.deconv1(features)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        
        # Final layer (no activation, will use softmax in loss)
        x = self.final_conv(x)
        return x

# Combine Encoder and Segmentation Decoder
class SegmentationModel(nn.Module):
    def __init__(self, pretrained_encoder, num_classes=3):
        super(SegmentationModel, self).__init__()
        self.encoder = pretrained_encoder
        self.decoder = SegmentationDecoder(num_classes)
        
        # Freeze the encoder weights (since it's pretrained)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Get features from the encoder (ignore latent)
        features, _ = self.encoder(x)
        # Use these features for segmentation
        segmentation = self.decoder(features)
        return segmentation

# Function to train the autoencoder
def train_autoencoder(model, train_loader, val_loader, num_epochs=10):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = next(model.parameters()).device
    
    train_losses = []
    val_losses = []
    
    # Initialize best validation loss to a large value
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, _ in pbar:
                images = images.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed, _, _ = model(images)
                loss = criterion(reconstructed, images)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item() * images.size(0)
                pbar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                reconstructed, _, _ = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'artifacts/best_autoencoder.pth')
            print(f"Saved best autoencoder model with validation loss: {best_val_loss:.4f}")
            
        # Visualize some reconstructions
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            visualize_reconstructions(model, val_loader, device, epoch)
            
    # Load the best model
    model.load_state_dict(torch.load('best_autoencoder.pth'))
    print(f"Loaded best autoencoder model with validation loss: {best_val_loss:.4f}")
            
    return train_losses, val_losses, best_val_loss

# Function to train the segmentation model
def train_segmentation(model, train_loader, val_loader, num_epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore white border pixels
    optimizer = optim.Adam(model.decoder.parameters(), lr=0.001)  # Only train decoder
    device = next(model.parameters()).device
    
    train_losses = []
    val_losses = []
    val_ious = []
    
    # Initialize best validation loss to a large value
    best_val_loss = float('inf')
    best_val_iou = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item() * images.size(0)
                pbar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        iou_scores = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                # Calculate IoU
                preds = torch.argmax(outputs, dim=1)
                iou = calculate_iou(preds, masks)
                iou_scores.append(iou)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        mean_iou = np.mean(iou_scores)
        val_ious.append(mean_iou)
        
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, IoU: {mean_iou:.4f}")
        
        # Save model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'artifacts/best_segmentation_val_loss.pth')
            print(f"Saved best segmentation model with validation loss: {best_val_loss:.4f}")
            
        # Also save model with best IoU (might be different from best loss)
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            torch.save(model.state_dict(), 'artifacts/best_segmentation_val_iou.pth')
            print(f"Saved best segmentation model with validation IoU: {best_val_iou:.4f}")
            
        # Visualize some segmentations
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            visualize_segmentations(model, val_loader, device, epoch)
            
    # Load the best model (by validation loss)
    model.load_state_dict(torch.load('best_segmentation_val_loss.pth'))
    print(f"Loaded best segmentation model with validation loss: {best_val_loss:.4f} and IoU: {best_val_iou:.4f}")
            
    return train_losses, val_losses, val_ious, best_val_loss, best_val_iou

# Helper function to calculate IoU
def calculate_iou(preds, targets, num_classes=3):
    ious = []
    
    # Ignore index 255 (white border)
    mask = (targets != 255)
    
    for cls in range(num_classes):
        pred_inds = (preds == cls) & mask
        target_inds = (targets == cls) & mask
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if union == 0:
            ious.append(float('nan'))  # Class not present
        else:
            ious.append(intersection / union)
    
    # Only consider classes that are present in the targets
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0

# Function to visualize reconstruction results
def visualize_reconstructions(model, data_loader, device, epoch):
    model.eval()
    with torch.no_grad():
        # Get a batch of images
        images, _ = next(iter(data_loader))
        images = images.to(device)
        
        # Get reconstructions
        reconstructed, _, _ = model(images)
        
        # Move tensors to CPU and convert to numpy arrays
        images = images.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        
        # Plot a few examples
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        for i in range(5):
            # Original images in top row
            axes[0, i].imshow(np.transpose(images[i], (1, 2, 0)))
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed images in bottom row
            axes[1, i].imshow(np.transpose(reconstructed[i], (1, 2, 0)))
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'artifacts/reconstruction_epoch_{epoch+1}.png')
        plt.close()

# Function to visualize segmentation results
def visualize_segmentations(model, data_loader, device, epoch):
    model.eval()
    
    # Class colors for visualization (RGB)
    colors = np.array([
        [0, 0, 0],       # Background (black)
        [255, 0, 0],     # Cat (red)
        [0, 255, 0]      # Dog (green)
    ])
    
    with torch.no_grad():
        # Get a batch of images
        images, masks = next(iter(data_loader))
        images = images.to(device)
        masks = masks.cpu().numpy()
        
        # Get segmentation predictions
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Move tensors to CPU and convert to numpy arrays
        images = images.cpu().numpy()
        
        # Plot a few examples
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        
        for i in range(5):
            # Original images in top row
            axes[0, i].imshow(np.transpose(images[i], (1, 2, 0)))
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Ground truth masks in middle row
            mask_rgb = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
            for j in range(3):
                mask_rgb[masks[i] == j] = colors[j]
            mask_rgb[masks[i] == 255] = [255, 255, 255]  # White for ignore index
            axes[1, i].imshow(mask_rgb)
            axes[1, i].set_title('Ground Truth')
            axes[1, i].axis('off')
            
            # Predicted masks in bottom row
            pred_rgb = np.zeros((*preds[i].shape, 3), dtype=np.uint8)
            for j in range(3):
                pred_rgb[preds[i] == j] = colors[j]
            axes[2, i].imshow(pred_rgb)
            axes[2, i].set_title('Prediction')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'artifacts/segmentation_epoch_{epoch+1}.png')
        plt.close()

# Main execution function
def run_experiment(train_loader, val_loader, test_loader, device, 
                   autoencoder_epochs=20, segmentation_epochs=20):
    # 1. Create and train the autoencoder
    print("Creating autoencoder model...")
    autoencoder = Autoencoder(in_channels=3, latent_dim=256).to(device)
    
    print("Training autoencoder...")
    autoencoder_train_losses, autoencoder_val_losses, best_ae_val_loss = train_autoencoder(
        autoencoder, train_loader, val_loader, num_epochs=autoencoder_epochs
    )
    
    # 2. Create segmentation model with pretrained encoder
    print("Creating segmentation model with pretrained encoder...")
    segmentation_model = SegmentationModel(
        autoencoder.encoder, num_classes=3  # 3 classes: background, cat, dog
    ).to(device)
    
    # 3. Train segmentation model
    print("Training segmentation model...")
    seg_train_losses, seg_val_losses, val_ious, best_seg_val_loss, best_val_iou = train_segmentation(
        segmentation_model, train_loader, val_loader, num_epochs=segmentation_epochs
    )
    
    # 4. Evaluate on test set
    print("Evaluating on test set with best model...")
    # First load the best model by IoU (usually more relevant for segmentation)
    segmentation_model.load_state_dict(torch.load('best_segmentation_val_iou.pth'))
    
    segmentation_model.eval()
    test_loss = 0.0
    test_ious = []
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = segmentation_model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)
            
            # Calculate IoU
            preds = torch.argmax(outputs, dim=1)
            iou = calculate_iou(preds, masks)
            test_ious.append(iou)
    
    test_loss = test_loss / len(test_loader.dataset)
    mean_test_iou = np.mean(test_ious)
    
    print(f"Test Loss: {test_loss:.4f}, Test IoU: {mean_test_iou:.4f}")
    print(f"Best validation loss was: {best_seg_val_loss:.4f}")
    print(f"Best validation IoU was: {best_val_iou:.4f}")
    
    # 6. Plot learning curves
    plt.figure(figsize=(12, 5))
    
    # Autoencoder loss
    plt.subplot(1, 2, 1)
    plt.plot(autoencoder_train_losses, label='Train Loss')
    plt.plot(autoencoder_val_losses, label='Val Loss')
    plt.axhline(y=best_ae_val_loss, color='r', linestyle='--', label=f'Best Val Loss: {best_ae_val_loss:.4f}')
    plt.title('Autoencoder Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Segmentation loss
    plt.subplot(1, 2, 2)
    plt.plot(seg_train_losses, label='Train Loss')
    plt.plot(seg_val_losses, label='Val Loss')
    plt.axhline(y=best_seg_val_loss, color='r', linestyle='--', label=f'Best Val Loss: {best_seg_val_loss:.4f}')
    plt.title('Segmentation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('artifacts/training_curves.png')
    
    # Plot IoU separately
    plt.figure(figsize=(6, 5))
    plt.plot(val_ious)
    plt.axhline(y=best_val_iou, color='r', linestyle='--', label=f'Best Val IoU: {best_val_iou:.4f}')
    plt.title('Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig('artifacts/validation_iou.png')
    
    # Save all results to a summary file
    with open('experiment_summary.txt', 'w') as f:
        f.write("Experiment Summary\n")
        f.write("=================\n\n")
        f.write(f"Autoencoder:\n")
        f.write(f"  - Best validation loss: {best_ae_val_loss:.4f}\n")
        f.write(f"\nSegmentation:\n")
        f.write(f"  - Best validation loss: {best_seg_val_loss:.4f}\n")
        f.write(f"  - Best validation IoU: {best_val_iou:.4f}\n")
        f.write(f"  - Test loss: {test_loss:.4f}\n")
        f.write(f"  - Test IoU: {mean_test_iou:.4f}\n")
    
    return {
        'autoencoder_train_losses': autoencoder_train_losses,
        'autoencoder_val_losses': autoencoder_val_losses,
        'best_ae_val_loss': best_ae_val_loss,
        'seg_train_losses': seg_train_losses,
        'seg_val_losses': seg_val_losses,
        'val_ious': val_ious,
        'best_seg_val_loss': best_seg_val_loss,
        'best_val_iou': best_val_iou,
        'test_loss': test_loss,
        'test_iou': mean_test_iou
    }

# Add this at the end of your script to execute
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Assuming train_loader, val_loader, and test_loader are already defined
    run_experiment(train_loader, val_loader, test_loader, 
                  device, 
                  autoencoder_epochs=5, 
                  segmentation_epochs=5)