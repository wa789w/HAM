import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
from tqdm import tqdm
import argparse

def evaluate_lpips_adversarial(original_dir, adversarial_dir, num_images=1000): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    
    # Transformation pipeline to resize and normalize images to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    lpips_scores = []
    
    print(f"Starting LPIPS calculation for {num_images} image pairs...")
    
    for i in tqdm(range(1, num_images + 1), desc="Evaluating LPIPS"):
        try:
            original_path = os.path.join(original_dir, f"{i}.png")
            adversarial_path = os.path.join(adversarial_dir, f"{i}.png")

            if not os.path.exists(original_path):
                print(f"Warning: Original image {original_path} not found. Skipping.")
                continue
            if not os.path.exists(adversarial_path):
                print(f"Warning: Adversarial image {adversarial_path} not found. Skipping.")
                continue

            original_img = Image.open(original_path).convert('RGB')
            adversarial_img = Image.open(adversarial_path).convert('RGB')
            
            original_tensor = transform(original_img).unsqueeze(0).to(device)
            adversarial_tensor = transform(adversarial_img).unsqueeze(0).to(device)

            with torch.no_grad():
                lpips_score = lpips_model(original_tensor, adversarial_tensor)
                lpips_scores.append(lpips_score.item())
                
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue

    lpips_scores = np.array(lpips_scores)
    mean_lpips = np.mean(lpips_scores) if len(lpips_scores) > 0 else 0
    
    print(f"\n=== LPIPS Evaluation Results ===")
    print(f"Number of successfully processed image pairs: {len(lpips_scores)}")
    print(f"Average LPIPS distance: {mean_lpips:.6f}")
    
    return lpips_scores, mean_lpips

def save_results(lpips_scores, output_file):
    """Saves the detailed LPIPS scores to a file."""
    try:
        with open(output_file, 'w') as f:
            f.write("Image_Pair\tLPIPS_Score\n")
            for i, score in enumerate(lpips_scores, 1):
                f.write(f"{i}\t{score:.6f}\n")
            
            if len(lpips_scores) > 0:
                f.write(f"\n=== Statistics ===\n")
                f.write(f"Mean: {np.mean(lpips_scores):.6f}\n")
        
        print(f"Detailed results have been saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LPIPS between original and adversarial images.")
    parser.add_argument('--original_dir', type=str, default="images", help='Directory containing the original images.')
    parser.add_argument('--adversarial_dir', type=str, default="output/adv", help='Directory containing the adversarial images.')                  
    parser.add_argument('--num_images', type=int, default=1000, help='Number of images to evaluate.')
    parser.add_argument('--output_file', type=str, default="lpips_results.txt", help='File to save the detailed LPIPS scores.')
    args = parser.parse_args()

    print(f"Original images directory: {args.original_dir}")
    print(f"Adversarial images directory: {args.adversarial_dir}")

    # Perform evaluation
    lpips_scores, mean_lpips = evaluate_lpips_adversarial(
        args.original_dir, args.adversarial_dir, args.num_images
    )
    
    # Save results if any scores were calculated
    if len(lpips_scores) > 0:
        save_results(lpips_scores, args.output_file)
    else:
        print("No images were processed. No results to save.")