import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from skimage.util import random_noise
from scipy.signal import convolve2d
from tqdm import tqdm

class PerturbedPetDataset(Dataset):
    """
    Dataset for perturbed test images with different levels of perturbations
    """
    def __init__(self, root_dir, perturbation_type=None, intensity_level=0, transform=None):
        """
        Args:
            root_dir (string): Directory with the dataset
            perturbation_type (string): Type of perturbation to apply
            intensity_level (int): Level of intensity for the perturbation (0-9)
            transform (callable, optional): Transform to be applied on the input image and mask
        """
        self.root_dir = root_dir
        self.perturbation_type = perturbation_type
        self.intensity_level = intensity_level
        self.transform = transform
        
        # Define intensity values for each perturbation
        self.intensity_values = {
            'gaussian_noise': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
            'gaussian_blur': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'contrast_increase': [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],
            'contrast_decrease': [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],
            'brightness_increase': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            'brightness_decrease': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            'occlusion': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            'salt_pepper': [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
        }
        
        # Only set up paths if root_dir is provided
        if root_dir is not None:
            # Set paths for images and masks
            self.image_dir = os.path.join(root_dir, 'test', 'color')
            self.mask_dir = os.path.join(root_dir, 'test', 'label')
            
            # Get all image file names
            if os.path.exists(self.image_dir):
                self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                        if os.path.isfile(os.path.join(self.image_dir, f))])
            else:
                print(f"Warning: Image directory {self.image_dir} does not exist")
                self.image_files = []
        else:
            # If no root_dir is provided, we're just accessing the intensity values
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
        
        # Convert to PIL Image
        seg_mask_pil = Image.fromarray(seg_mask)
        
        # Apply perturbation if specified
        if self.perturbation_type is not None and self.intensity_level > 0:
            image = self.apply_perturbation(image)
        
        # Resize and convert to tensor
        image = TF.resize(image, (224, 224))
        seg_mask_pil = TF.resize(seg_mask_pil, (224, 224), interpolation=TF.InterpolationMode.NEAREST)
        
        image_tensor = TF.to_tensor(image)
        seg_mask_tensor = torch.from_numpy(np.array(seg_mask_pil)).long()
        
        return image_tensor, seg_mask_tensor, img_name
    
    def apply_perturbation(self, image):
        """Apply specified perturbation with given intensity level to the image"""
        img_np = np.array(image)
        
        if self.perturbation_type == 'gaussian_noise':
            # a) Gaussian pixel noise
            std_dev = self.intensity_values[self.perturbation_type][self.intensity_level]
            noise = np.random.normal(0, std_dev, img_np.shape).astype(np.int16)
            img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        elif self.perturbation_type == 'gaussian_blur':
            # b) Gaussian blurring with 3x3 mask
            blur_times = self.intensity_values[self.perturbation_type][self.intensity_level]
            
            # Create 3x3 Gaussian kernel
            kernel = np.array([[1, 2, 1], 
                              [2, 4, 2], 
                              [1, 2, 1]]) / 16.0
            
            # Apply convolution multiple times
            blurred = img_np.copy()
            if blur_times > 0:
                # Apply to each channel
                for _ in range(blur_times):
                    for c in range(3):
                        blurred[:,:,c] = convolve2d(blurred[:,:,c], kernel, mode='same', boundary='symm')
            
            img_np = blurred
            
        elif self.perturbation_type == 'contrast_increase':
            # c) Image Contrast Increase
            factor = self.intensity_values[self.perturbation_type][self.intensity_level]
            img_np = np.clip((img_np.astype(np.float32) * factor), 0, 255).astype(np.uint8)
            
        elif self.perturbation_type == 'contrast_decrease':
            # d) Image Contrast Decrease
            factor = self.intensity_values[self.perturbation_type][self.intensity_level]
            img_np = np.clip((img_np.astype(np.float32) * factor), 0, 255).astype(np.uint8)
            
        elif self.perturbation_type == 'brightness_increase':
            # e) Image Brightness Increase
            value = self.intensity_values[self.perturbation_type][self.intensity_level]
            img_np = np.clip((img_np.astype(np.int16) + value), 0, 255).astype(np.uint8)
            
        elif self.perturbation_type == 'brightness_decrease':
            # f) Image Brightness Decrease
            value = self.intensity_values[self.perturbation_type][self.intensity_level]
            img_np = np.clip((img_np.astype(np.int16) - value), 0, 255).astype(np.uint8)
            
        elif self.perturbation_type == 'occlusion':
            # g) Occlusion of the Image
            size = self.intensity_values[self.perturbation_type][self.intensity_level]
            if size > 0:
                h, w = img_np.shape[:2]
                # Random position for the occlusion
                x0 = np.random.randint(0, w - size) if w > size else 0
                y0 = np.random.randint(0, h - size) if h > size else 0
                
                # Create black square
                img_np[y0:y0+size, x0:x0+size, :] = 0
                
        elif self.perturbation_type == 'salt_pepper':
            # h) Salt and Pepper Noise
            amount = self.intensity_values[self.perturbation_type][self.intensity_level]
            if amount > 0:
                img_np = random_noise(img_np, mode='s&p', amount=amount, salt_vs_pepper=0.5)
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                
        return Image.fromarray(img_np)


def dice_coefficient(pred, target, ignore_index=255):
    """Calculate Dice coefficient (F1 score) for segmentation results"""
    # Convert predictions to class indices
    pred = pred.argmax(dim=1)
    
    # Create mask for valid pixels (not ignore_index)
    mask = (target != ignore_index)
    
    # Calculate Dice for each class
    dice_scores = []
    num_classes = pred.max().item() + 1
    
    for cls in range(num_classes):
        pred_cls = (pred == cls) & mask
        target_cls = (target == cls) & mask
        
        intersection = (pred_cls & target_cls).sum().float()
        union = pred_cls.sum().float() + target_cls.sum().float()
        
        # Avoid division by zero
        if union > 0:
            dice = (2 * intersection) / union
        else:
            dice = torch.tensor(1.0)  # If no pixels of this class, consider it perfect match
            
        dice_scores.append(dice.item())
    
    return np.mean(dice_scores)


def evaluate_model_on_perturbations(model, dataset_root, device, batch_size=8):
    """
    Evaluate model on all perturbations and their intensity levels.
    Returns a dictionary of dice scores for each perturbation and intensity.
    """
    perturbation_types = [
        'gaussian_noise', 'gaussian_blur', 'contrast_increase', 'contrast_decrease',
        'brightness_increase', 'brightness_decrease', 'occlusion', 'salt_pepper'
    ]
    
    results = {}
    
    # Set model to evaluation mode
    model.eval()
    
    for perturbation in perturbation_types:
        print(f"Evaluating {perturbation}...")
        perturbation_results = []
        
        # Evaluate for each intensity level
        for level in range(10):  # 10 intensity levels (0-9)
            dataset = PerturbedPetDataset(dataset_root, perturbation_type=perturbation, intensity_level=level)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            dice_scores = []
            
            with torch.no_grad():
                for images, masks, _ in tqdm(dataloader, desc=f"Level {level}"):
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    
                    # Calculate Dice coefficient
                    batch_dice = dice_coefficient(outputs, masks)
                    dice_scores.append(batch_dice)
            
            # Average Dice score for this intensity level
            avg_dice = np.mean(dice_scores)
            perturbation_results.append(avg_dice)
            print(f"  Level {level}: Dice = {avg_dice:.4f}")
            
        results[perturbation] = perturbation_results
    
    return results


def save_example_perturbations(dataset_root, output_dir):
    """
    Save example images for each perturbation and intensity level.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    perturbation_types = [
        'gaussian_noise', 'gaussian_blur', 'contrast_increase', 'contrast_decrease',
        'brightness_increase', 'brightness_decrease', 'occlusion', 'salt_pepper'
    ]
    
    # Use the first image from the test set for all examples
    img_name = None
    
    for perturbation in perturbation_types:
        perturbation_dir = os.path.join(output_dir, perturbation)
        os.makedirs(perturbation_dir, exist_ok=True)
        
        # Get the first image name if not already retrieved
        if img_name is None:
            test_dataset = PerturbedPetDataset(dataset_root)
            if len(test_dataset) > 0:
                _, _, img_name = test_dataset[0]
        
        # Generate example for each intensity level
        for level in range(10):
            dataset = PerturbedPetDataset(dataset_root, perturbation_type=perturbation, intensity_level=level)
            
            # Find the same image in the dataset
            for i in range(len(dataset)):
                img_tensor, _, curr_img_name = dataset[i]
                if curr_img_name == img_name:
                    # Convert tensor to PIL image for saving
                    img_pil = TF.to_pil_image(img_tensor)
                    
                    # Save the image
                    img_path = os.path.join(perturbation_dir, f"level_{level}.jpg")
                    img_pil.save(img_path)
                    break


def plot_perturbation_results(results, output_dir):
    """
    Plot the Dice scores vs. perturbation intensity for each perturbation type.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot each perturbation separately
    for perturbation, scores in results.items():
        plt.figure(figsize=(10, 6))
        
        # Get the intensity values for this perturbation
        dataset = PerturbedPetDataset(None)  # Dummy dataset to access intensity values
        intensity_values = dataset.intensity_values[perturbation]
        
        plt.plot(intensity_values, scores, 'o-', linewidth=2, markersize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.title(f"Segmentation Performance vs. {perturbation.replace('_', ' ').title()} Intensity", fontsize=14)
        plt.xlabel("Perturbation Intensity", fontsize=12)
        plt.ylabel("Mean Dice Score", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{perturbation}.png"), dpi=300)
        plt.close()
    
    # Also create a combined plot
    plt.figure(figsize=(12, 8))
    
    for perturbation, scores in results.items():
        # Get the intensity values for this perturbation
        intensity_level = list(range(len(scores)))
        
        plt.plot(intensity_level, scores, 'o-', linewidth=2, markersize=6, 
                 label=perturbation.replace('_', ' ').title())
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("Segmentation Performance vs. Perturbation Intensity", fontsize=14)
    plt.xlabel("Intensity Level", fontsize=12)
    plt.ylabel("Mean Dice Score", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_results.png"), dpi=300)
    plt.close()


def visualize_predictions(model, dataset_root, perturbation_type, levels, device, output_dir):
    """
    Generate and save side-by-side visualizations of the original image,
    ground truth, and model prediction for different perturbation levels.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a test image 
    test_dataset = PerturbedPetDataset(dataset_root)
    if len(test_dataset) == 0:
        print("No test images found.")
        return
    
    img_tensor, mask_tensor, img_name = test_dataset[0]
    
    # Create a color map for segmentation visualisation
    cmap = np.array([
        [0, 0, 0],        # Background (black)
        [128, 0, 0],      # Cat (red)
        [0, 128, 0]       # Dog (green)
    ])
    
    # For each level, generate visualisation
    for level in levels:
        # Get perturbed image
        perturbed_dataset = PerturbedPetDataset(
            dataset_root, 
            perturbation_type=perturbation_type, 
            intensity_level=level
        )
        
        # Find the same image
        for i in range(len(perturbed_dataset)):
            perturbed_img, perturbed_mask, curr_name = perturbed_dataset[i]
            if curr_name == img_name:
                # Make prediction
                model.eval()
                with torch.no_grad():
                    perturbed_img_tensor = perturbed_img.unsqueeze(0).to(device)
                    output = model(perturbed_img_tensor)
                    pred = output.argmax(dim=1)[0].cpu().numpy()
                
                # Convert to images for visualisation
                img_np = perturbed_img.permute(1, 2, 0).numpy()
                mask_np = perturbed_mask.numpy()
                
                # Create RGB mask images
                mask_rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
                pred_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
                
                for cls in range(3):  # 3 classes
                    mask_rgb[mask_np == cls] = cmap[cls]
                    pred_rgb[pred == cls] = cmap[cls]
                
                # White for ignore regions
                mask_rgb[mask_np == 255] = [255, 255, 255]
                
                # Create figure
                plt.figure(figsize=(15, 5))
                
                # Plot image, ground truth, and prediction
                plt.subplot(1, 3, 1)
                plt.imshow(img_np)
                plt.title("Perturbed Image", fontsize=12)
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(mask_rgb)
                plt.title("Ground Truth", fontsize=12)
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred_rgb)
                plt.title("Prediction", fontsize=12)
                plt.axis('off')
                
                # Add overall title
                intensity_value = perturbed_dataset.intensity_values[perturbation_type][level]
                plt.suptitle(f"{perturbation_type.replace('_', ' ').title()} (Level {level}, Value: {intensity_value})", 
                           fontsize=14)
                
                # Save figure
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(os.path.join(output_dir, f"{perturbation_type}_level_{level}.png"), dpi=300)
                plt.close()
                
                break


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Robustness evaluation for segmentation models')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, default='robustness_results', 
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--save_examples', action='store_true', 
                        help='Save example images for each perturbation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model_path).to(device)
    
    # Evaluate model on perturbations
    results = evaluate_model_on_perturbations(
        model, args.dataset_root, device, batch_size=args.batch_size
    )
    
    # Save the results as a numpy array for later use
    np.save(os.path.join(args.output_dir, 'perturbation_results.npy'), results)
    
    # Plot the results
    plot_perturbation_results(results, args.output_dir)
    
    # Save example perturbations if requested
    if args.save_examples:
        save_example_perturbations(args.dataset_root, os.path.join(args.output_dir, 'examples'))
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")
