import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import timm  # 添加timm库
from PIL import Image
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

    def evaluate_folder(self, image_folder, label_path, output_file=None, image_suffix="_adv.png"):
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
        # 计算攻击成功率
        attack_success_rate = 1 - (correct / total)

        # 写入总结统计
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
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet50', 'inception_v3', 'vgg16', 'densenet121'],
                        help='Model to use for evaluation')
    parser.add_argument('--output', type=str, default=None, help='Path to save results')
    parser.add_argument('--image_suffix', type=str, default="_adv.png", 
                        help='Suffix of image files (e.g., "_adv.png")')

    args = parser.parse_args()


if __name__ == "__main__":
    # 指定路径
    image_folder = "output_clip_1011/adv"
    output_path = 'evaluate_clip_1011'

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
    
    # --- START: 新增代码 ---
    # 1. 初始化一个字典来存储每张图片成功攻击的模型数量
    # 键是图片文件名，值是成功攻击的次数
    image_attack_counts = {}
    # --- END: 新增代码 ---

    # 创建一个文件来存储所有模型的准确率
    # 创建一个文件来存储所有模型的攻击成功率
    attack_summary_file = os.path.join(output_path, 'attack_summary.txt')
    with open(attack_summary_file, 'w') as summary_file:
        # 写入时间戳和用户信息
        summary_file.write("Date and Time (UTC): 2025-01-19 09:45:02\n")
        summary_file.write("User: wa789w\n")
        summary_file.write("\nModel Attack Success Rates:\n")
        summary_file.write("-" * 50 + "\n")
        summary_file.write("Model Name\tAttackSuccessRate\tSuccess/Total\n")
        summary_file.write("-" * 50 + "\n")

        total_attack_success_rate = 0.0  # 新增，统计所有模型攻击成功率之和

        for model_name in models_to_evaluate:
            try:
                print(f"\nEvaluating {model_name}...")
                evaluator = ImageClassificationEvaluator(model_name=model_name)

                label_path = "labels.txt"
                # 将输出文件路径指向 evaluate 文件夹
                output_file = os.path.join(output_path, f"{model_name}.txt")
                image_suffix = "_adv_image.png"

                # 运行评估
                attack_success_rate, results = evaluator.evaluate_folder(
                    image_folder=image_folder,
                    label_path=label_path,
                    output_file=output_file,
                    image_suffix=image_suffix
                )
                
                # 将攻击成功率写入汇总文件
                success = int(attack_success_rate * len(results))
                total = len(results)
                summary_file.write(f"{model_name}\t{attack_success_rate:.4f}\t{success}/{total}\n")

                total_attack_success_rate += attack_success_rate  # 累加总攻击成功率

                # --- START: 新增代码 ---
                # 2. 遍历单个模型的结果，更新每张图片的攻击成功计数
                for res in results:
                    image_file = res['image']
                    true_label = res['true_label']
                    predicted_label = res['predicted_label']

                    # 如果是第一次见到这张图片，先在字典中初始化
                    if image_file not in image_attack_counts:
                        image_attack_counts[image_file] = 0
                    
                    # 如果攻击成功 (预测标签 ≠ 真实标签)，则计数加一
                    if predicted_label != true_label:
                        image_attack_counts[image_file] += 1
                # --- END: 新增代码 ---
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                summary_file.write(f"{model_name}\tError: {str(e)}\n")
                continue

        # 写入攻击成功率总和
        summary_file.write("-" * 50 + "\n")
        summary_file.write(f"Sum of Attack Success Rates: {total_attack_success_rate:.4f}\n")
    
    # --- START: 新增代码 ---
    # 3. 在所有模型评估完成后，筛选出高度成功的图片并保存到 CSV
    print("\n" + "="*50)
    print("Finding highly successful adversarial images...")

    num_models = len(models_to_evaluate)
    # 设置成功率阈值为 80%
    success_threshold = 0.80
    
    highly_successful_images_indices = []
    
    # 遍历统计结果
    for image_file, success_count in image_attack_counts.items():
        # 计算该图片在所有模型上的攻击成功率
        attack_rate_for_image = success_count / num_models
        
        # 如果成功率 >= 阈值，则记录其序号
        if attack_rate_for_image >= success_threshold:
            # 从文件名（如 '1_adv_image.png' 或 '1.png'）中提取序号 '1'
            # 这种方法比较稳健，可以处理多种命名格式
            image_index = os.path.splitext(image_file)[0].split('_')[0]
            highly_successful_images_indices.append(int(image_index))
            
    # 对序号进行排序
    highly_successful_images_indices.sort()
    
    # 将结果保存到 CSV 文件
    output_csv_path = os.path.join(output_path, 'highly_successful_images.csv')
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvfile.write("Image_Index\n") # 写入表头
        for index in highly_successful_images_indices:
            csvfile.write(f"{index}\n")

    print(f"Found {len(highly_successful_images_indices)} images that succeeded on >= {success_threshold:.0%} of models.")
    print(f"Their indices have been saved to: {output_csv_path}")
    print("="*50)
    # --- END: 新增代码 ---