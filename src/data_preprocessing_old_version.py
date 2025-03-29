import os
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates


class JointTransform:
    def __init__(self, img_size=128, crop_size=112, p_flip=0.5, p_rotate=0.3, p_elastic=0.3):
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
            image = TF.center_crop(image, (self.img_size, self.img_size))
            mask = TF.center_crop(mask, (self.img_size, self.img_size))
        
        # Elastic transformation (only if probability threshold met)
        if random.random() < self.p_elastic:
            image, mask = self._elastic_transform(image, mask)
        
        # Color jitter (image only)
        image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        
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
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with the dataset.
            split (string): 'train', 'val', or 'test' split.
            transform (callable, optional): Transform to be applied on the input image.
            target_transform (callable, optional): Transform to be applied on the mask.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Set paths for images and masks
        self.image_dir = os.path.join(root_dir, split, 'color')
        self.mask_dir = os.path.join(root_dir, split, 'label')
        
        # Get all image file names
        self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                  if os.path.isfile(os.path.join(self.image_dir, f))])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Get corresponding mask file
        mask_name = img_name.replace('.jpg', '.png')
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
            image, seg_mask = self.transform(image, seg_mask_pil)
        else:
            # Convert to tensor if no transform
            image = TF.to_tensor(Image.fromarray(np.array(image)))
            seg_mask = torch.from_numpy(seg_mask).long()
        
        return image, seg_mask