import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
from tqdm import tqdm
import csv

def evaluate_lpips_adversarial(original_dir, adversarial_dir, num_images=1000):
    """
    Evaluate the LPIPS distance between original images and adversarial samples.
    
    Args:
        original_dir: Directory of original images.
        adversarial_dir: Directory of adversarial samples.
        num_images: Number of image pairs to evaluate.
    
    Returns:
        lpips_scores: LPIPS score for each image pair.
        mean_lpips: Mean LPIPS score.
    """
    
    # Initialize the LPIPS model with the VGG backbone.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    
    # LPIPS expects inputs in the [-1, 1] range.
    transform_original = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transform_adversarial = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    lpips_scores = []
    
    print(f"Computing LPIPS distance for {num_images} image pairs...")
    
    for i in tqdm(range(1, num_images + 1)):
        try:
            original_path = os.path.join(original_dir, f"{i}.png")
            adversarial_path = os.path.join(adversarial_dir, f"{i}.png")
            
            if not os.path.exists(original_path):
                print(f"Warning: original image does not exist: {original_path}")
                continue
            if not os.path.exists(adversarial_path):
                print(f"Warning: adversarial sample does not exist: {adversarial_path}")
                continue
            
            original_img = Image.open(original_path).convert('RGB')
            adversarial_img = Image.open(adversarial_path).convert('RGB')
            
            original_tensor = transform_original(original_img).unsqueeze(0).to(device)
            adversarial_tensor = transform_adversarial(adversarial_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                lpips_score = lpips_model(original_tensor, adversarial_tensor)
                lpips_scores.append(lpips_score.item())
                
        except Exception as e:
            print(f"Error while processing image {i}: {e}")
            continue
    
    lpips_scores = np.array(lpips_scores)
    mean_lpips = np.mean(lpips_scores)
    std_lpips = np.std(lpips_scores)
    min_lpips = np.min(lpips_scores)
    max_lpips = np.max(lpips_scores)
    
    print(f"\n=== LPIPS Evaluation Results ===")
    print(f"Successfully processed image pairs: {len(lpips_scores)}")
    print(f"Mean LPIPS distance: {mean_lpips:.6f}")
    print(f"Standard deviation: {std_lpips:.6f}")
    print(f"Minimum: {min_lpips:.6f}")
    print(f"Maximum: {max_lpips:.6f}")
    
    return lpips_scores, mean_lpips

def save_results(lpips_scores, output_file="lpips_results.txt"):
    """Save detailed LPIPS results to a file."""
    with open(output_file, 'w') as f:
        f.write("Image_Pair\tLPIPS_Score\n")
        for i, score in enumerate(lpips_scores, 1):
            f.write(f"{i}\t{score:.6f}\n")
        
        f.write(f"\n=== Statistics ===\n")
        f.write(f"Mean: {np.mean(lpips_scores):.6f}\n")
        f.write(f"Standard deviation: {np.std(lpips_scores):.6f}\n")
        f.write(f"Minimum: {np.min(lpips_scores):.6f}\n")
        f.write(f"Maximum: {np.max(lpips_scores):.6f}\n")
    
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    original_dir = "images"
    adversarial_dir = "output_clip_1011/adv"
    print(adversarial_dir)
    
    lpips_scores, mean_lpips = evaluate_lpips_adversarial(
        original_dir, adversarial_dir, num_images=1000
    )
    
    save_results(lpips_scores, "lpips_results.txt")
    
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(lpips_scores, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('LPIPS Score')
        plt.ylabel('Frequency')
        plt.title(f'LPIPS Score Distribution (Mean: {mean_lpips:.6f})')
        plt.axvline(mean_lpips, color='red', linestyle='--', label=f'Mean: {mean_lpips:.6f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('lpips_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("LPIPS distribution plot saved as: lpips_distribution.png")
        
    except ImportError:
        print("matplotlib is not installed; skipping distribution plot.")

