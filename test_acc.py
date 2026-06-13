import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import timm  # 添加timm库
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import argparse
# from model_SIN import load_model

model_A = "resnet50_trained_on_SIN"
model_B = "resnet50_trained_on_SIN_and_IN"
model_C = "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN"

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
        """加载指定的预训练模型"""
        if self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif self.model_name == 'vgg19':
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        elif self.model_name == 'inception_v3':
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        elif self.model_name == 'convnext':
            model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        elif self.model_name == 'vit_b_16':
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif self.model_name == 'swin_b':
            model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        # 使用timm加载DeiT和Mixer模型
        elif self.model_name == 'deit_b':
            model = timm.create_model('deit_base_patch16_224', pretrained=True)
        elif self.model_name == 'deit_s':
            model = timm.create_model('deit_small_patch16_224', pretrained=True)
        elif self.model_name == 'mixer_b_16':
            model = timm.create_model('mixer_b16_224', pretrained=True)
        elif self.model_name == 'mixer_l_16':
            model = timm.create_model('mixer_l16_224', pretrained=True)
        # 在现有代码中添加这些模型
        elif self.model_name == 'vit_l_16':
            model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        elif self.model_name == 'vit_l_32':
            model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
        elif self.model_name == 'vit_s_16':
            model = models.vit_s_16(weights=models.ViT_S_16_Weights.IMAGENET1K_V1)
        elif self.model_name == 'vit_h_14':
            model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_V1)
        # elif self.model_name == 'model_sin':
        #     model = load_model(model_A)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transform(self):
        """获取模型对应的图像预处理转换"""
        if self.model_name == 'inception_v3':
            input_size = 299
        else:
            input_size = 224  # 其他所有模型都使用 224x224

        return transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image):
        """对单张图像进行预测"""
        with torch.no_grad():
            output = self.model(image)
            if isinstance(output, tuple):  # 处理inception_v3的辅助输出
                output = output[0]
            
            # 获取预测结果
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top1_prob, top1_class = torch.max(probabilities, 1)
            
            return top1_class.item(), top1_prob.item()

    def evaluate_folder(self, image_folder, label_path, output_file=None, image_suffix="_adv.png", resize_before=None, jpeg_compress=False, jpeg_quality=None):
        """评估文件夹中的图像"""
        # 读取标签
        with open(label_path, 'r') as f:
            true_labels = [int(line.strip()) - 1 for line in f.readlines()]  # 减1处理

        # 获取图像文件列表
        image_files = []
        for i in range(1, len(true_labels) + 1):
            index = i-1
            # 支持不同的文件命名格式
            possible_names = [
                f"{i}.png",           # 原始格式
                f"{i}{image_suffix}", # 带后缀格式（如 1_adv.png）
                f"{index:04}_adv_image.png"
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

        # 确保图像和标签数量匹配
        assert len(image_files) == len(true_labels), "Number of images does not match number of labels"

        correct = 0
        total = len(image_files)
        results = []

        # 创建输出文件
        if output_file:
            f_out = open(output_file, 'w')
            f_out.write("Image\tTrue_Label\tPredicted_Label\tConfidence\n")

        # 处理每张图像
        for img_idx, image_file in enumerate(tqdm(image_files, desc=f"Evaluating with {self.model_name}")):
            try:
                # 加载和预处理图像
                image_path = os.path.join(image_folder, image_file)
                image = Image.open(image_path).convert('RGB')
                if resize_before:
                    image = image.resize(resize_before, Image.Resampling.LANCZOS)
                if jpeg_compress:
                    buffered = BytesIO()
                    save_kwargs = {}
                    if jpeg_quality is not None:
                        save_kwargs['quality'] = jpeg_quality
                    image.save(buffered, format='JPEG', **save_kwargs)
                    buffered.seek(0)
                    image = Image.open(buffered).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)

                # 预测
                pred_class, pred_prob = self.predict_image(image_tensor)

                # 更新统计
                true_label = true_labels[img_idx]
                correct += (pred_class == true_label)

                # 保存结果
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

        # 计算准确率
        accuracy = correct / total

        # 写入总结统计
        if output_file:
            f_out.write(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
            f_out.close()

        print(f"\nEvaluation Results for {self.model_name}:")
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

        return accuracy, results

def main():
    parser = argparse.ArgumentParser(description='Evaluate image classification accuracy')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the image folder')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the label file')
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet50', 'inception_v3', 'vgg16', 'densenet121'],
                        help='Model to use for evaluation')
    parser.add_argument('--output', type=str, default=None, help='Path to save results')
    parser.add_argument('--image_suffix', type=str, default="_adv.png", 
                        help='Suffix of image files (e.g., "_adv.png")')

    args = parser.parse_args()


if __name__ == "__main__":
    # 指定路径
    image_folder = "output_epsilon_33/adv"
    output_path = 'evaluate_epsilon_33_64jpeg'

    # 创建 evaluate 文件夹
    os.makedirs(output_path, exist_ok=True)
    
    # 创建所有模型的评估器并运行评估
    models_to_evaluate = [
        # # CNN架构
        'resnet50', 'vgg19', 'mobilenet_v2', 'inception_v3',
        # 'convnext'
        # 'model_sin'
        # # Transformer架构
        'vit_b_16', 'swin_b', 'deit_b', 'deit_s',
        # # MLP架构
        'mixer_b_16', 'mixer_l_16'
    ]
    
    # 创建一个文件来存储所有模型的准确率
    accuracy_summary_file = os.path.join(output_path, 'accuracy_summary.txt')
    with open(accuracy_summary_file, 'w') as summary_file:
        # 写入时间戳和用户信息
        summary_file.write("Date and Time (UTC): 2025-01-19 09:45:02\n")
        summary_file.write("User: wa789w\n")
        summary_file.write("\nModel Accuracies:\n")
        summary_file.write("-" * 50 + "\n")
        summary_file.write("Model Name\tAccuracy\tCorrect/Total\n")
        summary_file.write("-" * 50 + "\n")
    
        for model_name in models_to_evaluate:
            try:
                print(f"\nEvaluating {model_name}...")
                evaluator = ImageClassificationEvaluator(model_name=model_name)

                label_path = "labels.txt"
                # 将输出文件路径指向 evaluate 文件夹
                output_file = os.path.join(output_path, f"{model_name}.txt")
                image_suffix = "_adv_image.png"

                # 运行评估
                accuracy, results = evaluator.evaluate_folder(
                    image_folder=image_folder,
                    label_path=label_path,
                    output_file=output_file,
                    image_suffix=image_suffix,
                    resize_before=(32, 32),
                    jpeg_compress=True
                )
                
                # 将准确率写入汇总文件
                correct = int(accuracy * len(results))
                total = len(results)
                summary_file.write(f"{model_name}\t{accuracy:.4f}\t{correct}/{total}\n")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                summary_file.write(f"{model_name}\tError: {str(e)}\n")
                continue
        
        # 写入结束分隔线
        summary_file.write("-" * 50 + "\n")