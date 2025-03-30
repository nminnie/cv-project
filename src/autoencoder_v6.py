import os
import sys
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


class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # Input: (3, 128, 128)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 64, 64)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 32, 32)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (128, 16, 16)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (256, 8, 8)
            
            nn.Conv2d(256, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (latent_dim, 4, 4)
        )
        
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # Input: (latent_dim, 4, 4)
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # (3, 128, 128)
            nn.Sigmoid()  # Output values between 0 and 1
        )
        
    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


class SegmentationDecoder(nn.Module):
    def __init__(self, latent_dim=256, num_classes=3):
        super(SegmentationDecoder, self).__init__()
        self.decoder = nn.Sequential(
            # Input: (latent_dim, 4, 4)
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1),  # (num_classes, 128, 128)
        )
        
    def forward(self, x):
        return self.decoder(x)


class SegmentationModel(nn.Module):
    def __init__(self, pretrained_encoder, latent_dim, num_classes=3):
        super(SegmentationModel, self).__init__()
        self.encoder = pretrained_encoder
        self.decoder = SegmentationDecoder(latent_dim, num_classes)
        
        # Freeze the encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Get features from the encoder (ignore latent)
        features = self.encoder(x)
        # Use these features for segmentation
        segmentation = self.decoder(features)
        return segmentation


def train_autoencoder(model, train_loader, val_loader, run_path, num_epochs=10):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = next(model.parameters()).device
    
    train_losses = []
    val_losses = []

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
                outputs = model(images)
                loss = criterion(outputs, images)
                
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
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{run_path}/best_autoencoder.pth')
            print(f"Saved best autoencoder model with validation loss: {best_val_loss:.4f}")
            
        # Visualize some reconstructions
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            visualize_reconstructions(model, val_loader, device, run_path, epoch)

    return train_losses, val_losses, best_val_loss


def train_segmentation(model, train_loader, val_loader, run_path, num_epochs=10):
    model.train()
    
    device = next(model.parameters()).device
    cat_weight = 2.1
    class_weights = torch.tensor([1.0, cat_weight, 1.0], device=device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)  # Ignore white border pixels
    optimizer = optim.Adam(model.decoder.parameters(), lr=0.001)  # Only train decoder
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
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
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{run_path}/best_segmentation.pth')
            print(f"Saved best segmentation model with validation loss: {best_val_loss:.4f}")
            
        # Visualize some segmentations
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            visualize_segmentations(model, val_loader, device, run_path, epoch)
            
    return train_losses, val_losses, best_val_loss


def evaluate_segmentation(model, dataloader, device):
    """
    Evaluate the segmentation model.

    Args:
        model: segmentation model
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    # Initialize metrics
    correct = 0
    total = 0

    # Initialize IoU metrics - one for each class
    num_classes = 3  # background (0), cat (1), dog (2)
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            # Get predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Calculate accuracy (ignoring white pixels with value 255)
            mask = (masks != 255)
            correct += (predicted[mask] == masks[mask]).sum().item()
            total += mask.sum().item()

            # Calculate IoU for each class
            for cls in range(num_classes):
                pred_cls = (predicted == cls) & mask
                true_cls = (masks == cls) & mask

                # Intersection and union
                intersection[cls] += (pred_cls & true_cls).sum().item()
                union[cls] += (pred_cls | true_cls).sum().item()

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0

    # Calculate IoU for each class
    class_names = ["background", "cat", "dog"]
    class_ious = []

    for cls in range(num_classes):
        iou = intersection[cls] / union[cls] if union[cls] > 0 else 0
        class_ious.append(iou)
        print(f"IoU for {class_names[cls]}: {iou:.4f}")

    # Calculate mean IoU
    mean_iou = sum(class_ious) / len(class_ious)

    print(f"Pixel Accuracy: {accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")

    return {
        "pixel_accuracy": accuracy,
        "class_ious": {class_names[i]: class_ious[i] for i in range(num_classes)},
        "mean_iou": mean_iou
    }


def visualize_reconstructions(model, data_loader, device, run_path, epoch):
    model.eval()
    with torch.no_grad():
        # Get a batch of images
        images, _ = next(iter(data_loader))
        images = images.to(device)
        
        # Get reconstructions
        reconstructed = model(images)
        
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
        plt.savefig(f'{run_path}/reconstruction_epoch_{epoch+1}.png')
        plt.close()


def visualize_segmentations(model, data_loader, device, run_path, epoch):
    model.eval()
    
    colors = np.array([
        [0, 0, 0],       # Background (black)
        [128, 0, 0],     # Cat (red)
        [0, 128, 0]      # Dog (green)
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
            pred_rgb[masks[i] == 255] = [255, 255, 255]  # White for ignore index
            axes[2, i].imshow(pred_rgb)
            axes[2, i].set_title('Prediction')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{run_path}/segmentation_epoch_{epoch+1}.png')
        plt.close()


def run_experiment(train_loader, val_loader, test_loader, device, run_path,
                   autoencoder_epochs=20, segmentation_epochs=20, latent_dim=256):
    
    # Train autoencoder
    print("Training autoencoder...")
    autoencoder = Autoencoder(latent_dim=latent_dim).to(device)
    autoencoder_train_losses, autoencoder_val_losses, best_ae_val_loss = train_autoencoder(
        autoencoder, train_loader, val_loader, run_path, num_epochs=autoencoder_epochs
    )
    
    # Train segmentation model
    print("Training segmentation model...")
    segmentation_model = SegmentationModel(autoencoder.encoder, latent_dim=latent_dim, num_classes=3).to(device)
    seg_train_losses, seg_val_losses, best_seg_val_loss = train_segmentation(
        segmentation_model, train_loader, val_loader, run_path, num_epochs=segmentation_epochs
    )
    
    # Evaluate segmentation model
    segmentation_model.load_state_dict(torch.load(f'{run_path}/best_segmentation.pth'))
    
    print("Validation set metrics:")
    val_metrics = evaluate_segmentation(segmentation_model, val_loader, device)

    print("Test set metrics:")
    test_metrics = evaluate_segmentation(segmentation_model, test_loader, device)
    
    # Plot training and validation losses
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
    plt.savefig(f'{run_path}/training_curves.png')

    # Save all results to a summary file
    with open(f'{run_path}/summary.txt', 'w') as f:
        f.write(f"Autoencoder:\n")
        f.write(f"  - Best validation loss: {best_ae_val_loss:.4f}\n")
        f.write(f"\nSegmentation:\n")
        f.write(f"  - Best validation loss: {best_seg_val_loss:.4f}\n")
        f.write(f"  - Validation set metrics: {val_metrics}\n")
        f.write(f"  - Test set metrics: {test_metrics}\n")
    
    return {
        'autoencoder_train_losses': autoencoder_train_losses,
        'autoencoder_val_losses': autoencoder_val_losses,
        'best_ae_val_loss': best_ae_val_loss,
        'seg_train_losses': seg_train_losses,
        'seg_val_losses': seg_val_losses,
        'best_seg_val_loss': best_seg_val_loss,
    }


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    run_name = sys.argv[1]
    run_path = f"./runs/{run_name}"
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    # Define image sizes
    img_size = 128
    crop_size = 128

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
    data_root = './Dataset/'
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
    
    run_experiment(train_loader, val_loader, test_loader, 
                  device, run_path,
                  autoencoder_epochs=20, 
                  segmentation_epochs=20)