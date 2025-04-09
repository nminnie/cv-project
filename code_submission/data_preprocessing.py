import os
import shutil
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates

seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JointTransform:
    def __init__(self, img_size=224, crop_size=196, p_flip=0.5, p_rotate=0.3, p_elastic=0.3):
        self.img_size = img_size
        self.crop_size = crop_size
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_elastic = p_elastic
        
        # For elastic transform
        self.elastic_alpha = 25
        self.elastic_sigma = 3
    
    def __call__(self, image, mask):
        # Resize both to same size
        image = TF.resize(image, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.NEAREST)
        
        # Random horizontal flip
        if random.random() < self.p_flip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            
        # Random rotation (-30 to 30 degrees)
        if random.random() < self.p_rotate:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            
        # Random crop (ensures consistent crop for both)
        i, j, h, w = transforms.RandomCrop.get_params(image, (self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        
        # Random translation (via padding + crop)
        pad_max = self.img_size // 10
        if pad_max > 0:
            pad_left = random.randint(0, pad_max)
            pad_top = random.randint(0, pad_max)
            pad_right = random.randint(0, pad_max)
            pad_bottom = random.randint(0, pad_max)
            image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom))
            mask = TF.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=255)
            
            # Crop back to original size
            image = TF.center_crop(image, (self.crop_size, self.crop_size))
            mask = TF.center_crop(mask, (self.crop_size, self.crop_size))
        
        # Elastic transformation (only if probability threshold met)
        if random.random() < self.p_elastic:
            image, mask = self._elastic_transform(image, mask)
        
        # Color jitter (image only)
        image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        
        # Final resize to 224x224 (ensures all images are the right size after transformations)
        image = TF.resize(image, (224, 224))
        mask = TF.resize(mask, (224, 224), interpolation=TF.InterpolationMode.NEAREST)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask
    
    def _elastic_transform(self, image, mask):
        """Apply elastic transformation on image and mask"""
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Create displacement fields
        shape = image_np.shape[:2]
        dx = np.random.rand(*shape) * 2 - 1
        dy = np.random.rand(*shape) * 2 - 1
        
        # Smooth displacement fields
        dx = gaussian_filter(dx, sigma=self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter(dy, sigma=self.elastic_sigma) * self.elastic_alpha
        
        # Create mesh grid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        
        # Apply displacement on image
        distorted_image = np.zeros(image_np.shape)
        for c in range(3):
            distorted_image[:,:,c] = map_coordinates(image_np[:,:,c], indices, order=1).reshape(shape)
        
        # Apply displacement on mask (using nearest to preserve labels)
        distorted_mask = map_coordinates(mask_np, indices, order=0).reshape(shape)
        
        return Image.fromarray(distorted_image.astype(np.uint8)), Image.fromarray(distorted_mask.astype(np.uint8))


class PetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with the dataset.
            split (string): 'train', 'val' or 'test' split.
            transform (callable, optional): Transform to be applied on the input image and mask.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Set paths for images and masks
        self.image_dir = os.path.join(root_dir, split, 'color')
        self.mask_dir = os.path.join(root_dir, split, 'label')
        
        # Get all image file names
        if os.path.exists(self.image_dir):
            self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                      if os.path.isfile(os.path.join(self.image_dir, f))])
        else:
            print(f"Warning: Image directory {self.image_dir} does not exist")
            self.image_files = []
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Get corresponding mask file
        mask_name = img_name.replace('.jpg', '.png')
        if not os.path.exists(os.path.join(self.mask_dir, mask_name)):
            # Try with other extensions if needed
            mask_name = img_name.split('.')[0] + '.png'
            
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        
        # Load and process mask
        mask = Image.open(mask_path).convert('RGB')
        mask_np = np.array(mask)
        
        # Convert RGB mask to class indices
        seg_mask = np.zeros(mask_np.shape[:2], dtype=np.uint8)
        
        # Background (black)
        black_mask = (mask_np[:,:,0] == 0) & (mask_np[:,:,1] == 0) & (mask_np[:,:,2] == 0)
        seg_mask[black_mask] = 0
        # Cat (red)
        red_mask = (mask_np[:,:,0] == 128) & (mask_np[:,:,1] == 0) & (mask_np[:,:,2] == 0)
        seg_mask[red_mask] = 1
        # Dog (green)
        green_mask = (mask_np[:,:,0] == 0) & (mask_np[:,:,1] == 128) & (mask_np[:,:,2] == 0)
        seg_mask[green_mask] = 2
        # White border (ignore)
        white_mask = (mask_np[:,:,0] == 255) & (mask_np[:,:,1] == 255) & (mask_np[:,:,2] == 255)
        seg_mask[white_mask] = 255  # Use 255 as ignore index
        
        # Create a PIL Image from the segmentation mask
        seg_mask_pil = Image.fromarray(seg_mask)
        
        # Apply joint transformations
        if self.transform:
            image_transformed, seg_mask_transformed = self.transform(image, seg_mask_pil)
            return image_transformed, seg_mask_transformed
        else:
            # Convert to tensor if no transform
            image = TF.resize(image, (224, 224))
            seg_mask_pil = TF.resize(seg_mask_pil, (224, 224), interpolation=TF.InterpolationMode.NEAREST)
            
            image_tensor = TF.to_tensor(image)
            seg_mask_tensor = torch.from_numpy(np.array(seg_mask_pil)).long()
            
            return image_tensor, seg_mask_tensor


def create_augmented_dataset(src_root, dst_root, augmentation_per_image=3, train_ratio=0.8):
    """
    Create an augmented dataset with transformed images and their masks.
    First creates an 80/20 train/val split, then applies transformations only to training images.
    
    Args:
        src_root (str): Source directory containing original dataset
        dst_root (str): Destination directory for augmented dataset
        augmentation_per_image (int): Number of augmented versions to create per original image
        train_ratio (float): Ratio of training data (default: 0.8 for 80/20 split)
    """
    # Define image sizes and transforms
    img_size = 224
    crop_size = 196  # Adjusted based on the original ratio (112/128)
    
    # Create joint transform for augmentation
    joint_transform = JointTransform(img_size=img_size, crop_size=crop_size)
    
    # Create destination directory structure if it doesn't exist
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dst_root, split, "color"), exist_ok=True)
        os.makedirs(os.path.join(dst_root, split, "label"), exist_ok=True)
    
    # First, handle the "TrainVal" set by splitting into train and val
    trainval_dataset = PetDataset(src_root, "TrainVal", transform=None)
    total_files = len(trainval_dataset.image_files)
    
    # Create train/val split indices
    train_size = int(train_ratio * total_files)
    val_size = total_files - train_size
    
    # Randomly shuffle indices for unbiased split
    all_indices = list(range(total_files))
    random.shuffle(all_indices)
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    
    print(f"Splitting TrainVal set: {train_size} training images, {val_size} validation images")
    
    # Process training set (with augmentation)
    print("Processing training set...")
    for i, idx in enumerate(train_indices):
        img_name = trainval_dataset.image_files[idx]
        src_img_path = os.path.join(trainval_dataset.image_dir, img_name)
        
        # Get corresponding mask file name
        mask_name = img_name.replace('.jpg', '.png')
        if not os.path.exists(os.path.join(trainval_dataset.mask_dir, mask_name)):
            mask_name = img_name.split('.')[0] + '.png'
            
        src_mask_path = os.path.join(trainval_dataset.mask_dir, mask_name)
        
        dst_img_path = os.path.join(dst_root, "train", "color", img_name)
        dst_mask_path = os.path.join(dst_root, "train", "label", mask_name)
        
        # Copy original files
        shutil.copy2(src_img_path, dst_img_path)
        shutil.copy2(src_mask_path, dst_mask_path)
        
        # Load original image and mask for augmentation
        image = Image.open(src_img_path).convert('RGB')
        mask = Image.open(src_mask_path).convert('RGB')
        
        # Process mask to create segmentation mask
        mask_np = np.array(mask)
        seg_mask = np.zeros(mask_np.shape[:2], dtype=np.uint8)
        
        # Background (black)
        black_mask = (mask_np[:,:,0] == 0) & (mask_np[:,:,1] == 0) & (mask_np[:,:,2] == 0)
        seg_mask[black_mask] = 0
        # Cat (red)
        red_mask = (mask_np[:,:,0] == 128) & (mask_np[:,:,1] == 0) & (mask_np[:,:,2] == 0)
        seg_mask[red_mask] = 1
        # Dog (green)
        green_mask = (mask_np[:,:,0] == 0) & (mask_np[:,:,1] == 128) & (mask_np[:,:,2] == 0)
        seg_mask[green_mask] = 2
        # White border (ignore)
        white_mask = (mask_np[:,:,0] == 255) & (mask_np[:,:,1] == 255) & (mask_np[:,:,2] == 255)
        seg_mask[white_mask] = 255  # Use 255 as ignore index
        
        seg_mask_pil = Image.fromarray(seg_mask)
        
        # Generate augmented versions (only for training set)
        for aug_idx in range(augmentation_per_image):
            # Create augmented image and mask
            aug_img, aug_mask_tensor = joint_transform(image, seg_mask_pil)
            
            # Convert tensor back to PIL for saving
            aug_img_pil = TF.to_pil_image(aug_img)
            aug_mask_np = aug_mask_tensor.numpy().astype(np.uint8)
            
            # Convert segmentation mask back to RGB for saving
            aug_mask_rgb = np.zeros((aug_mask_np.shape[0], aug_mask_np.shape[1], 3), dtype=np.uint8)
            # Background
            aug_mask_rgb[aug_mask_np == 0] = [0, 0, 0]
            # Cat
            aug_mask_rgb[aug_mask_np == 1] = [128, 0, 0]
            # Dog
            aug_mask_rgb[aug_mask_np == 2] = [0, 128, 0]
            # Ignore
            aug_mask_rgb[aug_mask_np == 255] = [255, 255, 255]
            
            aug_mask_pil = Image.fromarray(aug_mask_rgb)
            
            # Save augmented files with unique names
            aug_img_name = f"{img_name.split('.')[0]}_aug{aug_idx}.jpg"
            aug_mask_name = f"{img_name.split('.')[0]}_aug{aug_idx}.png"
            
            aug_img_path = os.path.join(dst_root, "train", "color", aug_img_name)
            aug_mask_path = os.path.join(dst_root, "train", "label", aug_mask_name)
            
            aug_img_pil.save(aug_img_path)
            aug_mask_pil.save(aug_mask_path)
        
        # Print progress
        if (i + 1) % 10 == 0 or i == len(train_indices) - 1:
            print(f"Processed {i + 1}/{len(train_indices)} training images")
    
    # Process validation set (no augmentation)
    print("Processing validation set...")
    for i, idx in enumerate(val_indices):
        img_name = trainval_dataset.image_files[idx]
        src_img_path = os.path.join(trainval_dataset.image_dir, img_name)
        
        # Get corresponding mask file name
        mask_name = img_name.replace('.jpg', '.png')
        if not os.path.exists(os.path.join(trainval_dataset.mask_dir, mask_name)):
            mask_name = img_name.split('.')[0] + '.png'
            
        src_mask_path = os.path.join(trainval_dataset.mask_dir, mask_name)
        
        dst_img_path = os.path.join(dst_root, "val", "color", img_name)
        dst_mask_path = os.path.join(dst_root, "val", "label", mask_name)
        
        # Copy original files (no augmentation for validation set)
        shutil.copy2(src_img_path, dst_img_path)
        shutil.copy2(src_mask_path, dst_mask_path)
        
        # Print progress
        if (i + 1) % 10 == 0 or i == len(val_indices) - 1:
            print(f"Processed {i + 1}/{len(val_indices)} validation images")
    
    # Process test set (no augmentation)
    print("Processing test set...")
    test_dataset = PetDataset(src_root, "Test", transform=None)
    for i, img_name in enumerate(test_dataset.image_files):
        src_img_path = os.path.join(test_dataset.image_dir, img_name)
        
        # Get corresponding mask file name
        mask_name = img_name.replace('.jpg', '.png')
        if not os.path.exists(os.path.join(test_dataset.mask_dir, mask_name)):
            mask_name = img_name.split('.')[0] + '.png'
            
        src_mask_path = os.path.join(test_dataset.mask_dir, mask_name)
        
        dst_img_path = os.path.join(dst_root, "test", "color", img_name)
        dst_mask_path = os.path.join(dst_root, "test", "label", mask_name)
        
        # Copy original files (no augmentation for test set)
        shutil.copy2(src_img_path, dst_img_path)
        shutil.copy2(src_mask_path, dst_mask_path)
        
        # Print progress
        if (i + 1) % 10 == 0 or i == len(test_dataset.image_files) - 1:
            print(f"Processed {i + 1}/{len(test_dataset.image_files)} test images")
    
    print("Dataset augmentation complete!")


def get_data_loaders(data_root, batch_size=16, num_workers=0):
    """
    Create data loaders for training, validation and test sets.
    
    Args:
        data_root (str): Root directory of the dataset
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    # Define transforms - only for training set during runtime
    joint_transform_train = JointTransform(img_size=224, crop_size=196)
    
    # Simple transform for validation/test (no augmentation)
    def val_transform(image, mask):
        # Resize both to final size
        image = TF.resize(image, (224, 224))
        mask = TF.resize(mask, (224, 224), interpolation=TF.InterpolationMode.NEAREST)
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask
    
    # Create datasets using the new directory structure
    train_dataset = PetDataset(data_root, 'train', transform=joint_transform_train)
    val_dataset = PetDataset(data_root, 'val', transform=val_transform)
    test_dataset = PetDataset(data_root, 'test', transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Print dataset sizes
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Define paths
    src_dataset_root = '../Dataset/'
    dst_dataset_root = '../Dataset_augmented/'
    
    # Create augmented dataset
    create_augmented_dataset(src_dataset_root, dst_dataset_root, augmentation_per_image=3)
    
    # Create data loaders from the augmented dataset
    train_loader, val_loader, test_loader = get_data_loaders(dst_dataset_root, batch_size=16)
    
    print("Data loaders created successfully!")