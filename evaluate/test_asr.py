import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import timm  # Add timm library
from PIL import Image
from tqdm import tqdm
import argparse

class ImageClassificationEvaluator:
    def __init__(self, model_name='resnet50', device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model_name = model_name.lower()
        self.model = self._load_model()
        self.transform = self._get_transform()
        
    def _load_model(self):
        """Load the specified pretrained model"""
        if self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif self.model_name == 'vgg19':
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        elif self.model_name == 'inception_v3':
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        elif self.model_name == 'vit_b_16':
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif self.model_name == 'swin_b':
            model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        # Load DeiT and Mixer models using timm
        elif self.model_name == 'deit_b':
            model = timm.create_model('deit_base_patch16_224', pretrained=True)
        elif self.model_name == 'deit_s':
            model = timm.create_model('deit_small_patch16_224', pretrained=True)
        elif self.model_name == 'mixer_b_16':
            model = timm.create_model('mixer_b16_224', pretrained=True)
        elif self.model_name == 'mixer_l_16':
            model = timm.create_model('mixer_l16_224', pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transform(self):
        """Get image preprocessing transform corresponding to the model"""
        if self.model_name == 'inception_v3':
            input_size = 299
        else:
            input_size = 224  

        return transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image):
        """Predict on a single image"""
        with torch.no_grad():
            output = self.model(image)
            if isinstance(output, tuple): 
                output = output[0]
            
            # Get prediction results
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top1_prob, top1_class = torch.max(probabilities, 1)
            
            return top1_class.item(), top1_prob.item()

    def evaluate_folder(self, image_folder, label_path, output_file=None, image_suffix=".png"):
        """Evaluate images in folder"""
        # Read labels
        with open(label_path, 'r') as f:
            true_labels = [int(line.strip()) - 1 for line in f.readlines()]  # Subtract 1 for processing

        # Get image file list
        image_files = []
        for i in range(1, len(true_labels) + 1):
            index = i-1
            # Support different file naming formats
            possible_names = [
                f"{i}{image_suffix}",           # Original format with suffix
            ]
            
            found = False
            for img_name in possible_names:
                if os.path.exists(os.path.join(image_folder, img_name)):
                    image_files.append(img_name)
                    found = True
                    break
                    
            if not found:
                raise FileNotFoundError(
                    f"No matching image found for index {i} in {image_folder}\n"
                    f"Tried: {possible_names}"
                )

        assert len(image_files) == len(true_labels), "Number of images does not match number of labels"

        correct = 0
        total = len(image_files)
        results = []

        if output_file:
            f_out = open(output_file, 'w')
            f_out.write("Image\tTrue_Label\tPredicted_Label\tConfidence\n")

        for img_idx, image_file in enumerate(tqdm(image_files, desc=f"Evaluating with {self.model_name}")):
            try:
                image_path = os.path.join(image_folder, image_file)
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)

                pred_class, pred_prob = self.predict_image(image_tensor)

                true_label = true_labels[img_idx]
                correct += (pred_class == true_label)

                if output_file:
                    f_out.write(f"{image_file}\t{true_label}\t{pred_class}\t{pred_prob:.4f}\n")

                results.append({
                    'image': image_file,
                    'true_label': true_label,
                    'predicted_label': pred_class,
                    'confidence': pred_prob
                })

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue

        attack_success_rate = 1 - (correct / total)

        if output_file:
            f_out.write(f"\nAttack Success Rate: {attack_success_rate:.4f} ({total-correct}/{total})")
            f_out.close()

        print(f"\nEvaluation Results for {self.model_name}:")
        print(f"Attack Success Rate: {attack_success_rate:.4f} ({total-correct}/{total})")

        return attack_success_rate, results

def main():
    parser = argparse.ArgumentParser(description='Evaluate image classification accuracy')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the image folder')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the label file')
    parser.add_argument('--output_dir', type=str, default='evaluate_results', help='Directory to save results')
    parser.add_argument('--image_suffix', type=str, default=".png", 
                        help='Suffix of image files (e.g., ".png")')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fixed list of models to evaluate
    models_to_evaluate = [
        'resnet50', 'vgg19', 'mobilenet_v2', 'inception_v3',
        'vit_b_16', 'swin_b', 'deit_b', 'deit_s',
        'mixer_b_16', 'mixer_l_16'
    ]
    
    attack_summary_file = os.path.join(args.output_dir, 'attack_summary.txt')
    with open(attack_summary_file, 'w') as summary_file:
        summary_file.write("Model\tASR\tSuccess/Total\n")
        
        for model_name in models_to_evaluate:
            try:
                print(f"\nEvaluating {model_name}...")
                evaluator = ImageClassificationEvaluator(model_name=model_name)

                output_file = os.path.join(args.output_dir, f"{model_name}.txt")

                # Run evaluation
                attack_success_rate, results = evaluator.evaluate_folder(
                    image_folder=args.image_folder,
                    label_path=args.label_path,
                    output_file=output_file,
                    image_suffix=args.image_suffix
                )

                success = int(attack_success_rate * len(results))
                total = len(results)
                summary_file.write(f"{model_name}\t{attack_success_rate:.4f}\t{success}/{total}\n")

                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                summary_file.write(f"{model_name}\tError: {str(e)}\n")
                continue
        
        print(f"\nEvaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()