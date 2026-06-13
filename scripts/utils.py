import argparse, os
import PIL
import matplotlib.pyplot as plt
import torch

def latent2image_with_grad(model, latents):
    """使用model.first_stage_model (VAE)直接解码潜变量，保持梯度流"""
    # 缩放因子
    latents = 1 / 0.18215 * latents
    # 直接使用VAE解码，而不是高级包装函数
    decoded = model.first_stage_model.decode(latents)
    # 标准化到[-1,1]范围
    return decoded

def encode_with_fixed_seed(model, image, seed=8888):
    """使用固定随机种子的VAE编码"""
    # 保存当前随机状态
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    # 设置固定种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 执行编码（不带generator参数）
    with torch.no_grad():
        latents = model.get_first_stage_encoding(model.encode_first_stage(image))
    # 恢复随机状态
    torch.set_rng_state(rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    return latents

def compute_latent_means_and_vars(latent_list, output_dir, image_idx=None, plot_separate=True):
    """Calculate and visualize both means and variances of latent representations at each step"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Calculate mean and variance for each latent representation
    latent_means = []
    latent_vars = []
    
    for i, latent in enumerate(latent_list):
        # Calculate mean and variance across all channels and dimensions
        mean_value = latent.mean().item()
        var_value = latent.var().item()
        
        latent_means.append(mean_value)
        latent_vars.append(var_value)
        print(f"Step {i}: Mean = {mean_value:.6f}, Variance = {var_value:.6f}")
    
    if plot_separate:
        # Plot mean curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(latent_means)), latent_means, marker='o', color='blue')
        plt.title('Forward Mean Change')
        plt.xlabel('Step')
        plt.ylabel('Mean')
        plt.grid(True)
        
        # Add value labels
        for i, mean_val in enumerate(latent_means):
            if i % 3 == 0 or abs(mean_val) > 0.1:  # Every 3 steps or large values
                plt.annotate(f"{mean_val:.4f}", 
                            (i, mean_val),
                            textcoords="offset points", 
                            xytext=(0,10),
                            ha='center')
        
        # Save mean curve
        if image_idx is not None:
            mean_curve_path = os.path.join(output_dir, f"latent_means_{image_idx}.png")
        else:
            mean_curve_path = os.path.join(output_dir, "latent_means.png")
        
        plt.savefig(mean_curve_path)
        print(f"Mean curve saved to: {mean_curve_path}")
        
        # Plot variance curve (separate plot)
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(latent_vars)), latent_vars, marker='o', color='green')
        plt.title('Forward Variance Change')
        plt.xlabel('Step')
        plt.ylabel('Variance')
        plt.grid(True)
        
        # Add value labels for variance
        for i, var_val in enumerate(latent_vars):
            if i % 3 == 0 or var_val > 0.05:  # Every 3 steps or large values
                plt.annotate(f"{var_val:.4f}", 
                            (i, var_val),
                            textcoords="offset points", 
                            xytext=(0,10),
                            ha='center')
        
        # Save variance curve
        if image_idx is not None:
            var_curve_path = os.path.join(output_dir, f"latent_vars_{image_idx}.png")
        else:
            var_curve_path = os.path.join(output_dir, "latent_vars.png")
        
        plt.savefig(var_curve_path)
        print(f"Variance curve saved to: {var_curve_path}")
        
        # Combined mean and variance in one plot (dual Y-axis)
        fig, ax1 = plt.figure(figsize=(12, 7)), plt.gca()
        
        # Mean on primary Y axis (left)
        ax1.plot(range(len(latent_means)), latent_means, marker='o', color='blue', label='Mean')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Mean', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        
        # Variance on secondary Y axis (right)
        ax2 = ax1.twinx()
        ax2.plot(range(len(latent_vars)), latent_vars, marker='s', color='green', label='Variance')
        ax2.set_ylabel('Variance', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Add title and legend
        plt.title('Forward Mean and Variance Change')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Save combined plot
        if image_idx is not None:
            combined_path = os.path.join(output_dir, f"latent_mean_var_combined_{image_idx}.png")
        else:
            combined_path = os.path.join(output_dir, "latent_mean_var_combined.png")
        
        plt.savefig(combined_path, bbox_inches='tight')
        print(f"Combined mean and variance plot saved to: {combined_path}")
    
    return latent_means, latent_vars

def compute_backward_latent_means_and_vars(latent_list, output_dir, image_idx=None, plot_separate=True):
    """Calculate and plot both means and variances for the backward diffusion process"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if latent variable list has content
    if len(latent_list) == 0:
        print("Warning: No latent variables available for calculation!")
        return [], []
    
    # Calculate means and variances
    latent_means = []
    latent_vars = []
    
    for i, latent in enumerate(latent_list):
        # Calculate mean and variance
        mean_value = latent.mean().item()
        var_value = latent.var().item()
        
        # Step number (from high to low)
        step_idx = len(latent_list) - i - 1
        
        # Print values for each step
        print(f"Backward step {step_idx}: Mean = {mean_value:.10f}, Variance = {var_value:.10f}")
        
        # Add to lists
        latent_means.append(mean_value)
        latent_vars.append(var_value)
    
    if plot_separate:
        # Prepare x-axis range (from high to low)
        step_range = range(len(latent_means)-1, -1, -1)  # Backward steps (high to low)
        
        # Plot mean curve
        plt.figure(figsize=(10, 6))
        plt.plot(step_range, latent_means, marker='o', color='red')
        plt.title('Backward Mean Change')
        plt.xlabel('Step')
        plt.ylabel('Mean')
        plt.grid(True)
        
        # Add value labels
        for i, mean_val in enumerate(latent_means):
            step_idx = len(latent_means) - i - 1  # Reverse numbering
            if i % 2 == 0 or abs(mean_val) > 0.1:  # Every other step or large values
                plt.annotate(f"{mean_val:.4f}", 
                            (step_idx, mean_val),
                            textcoords="offset points", 
                            xytext=(0,10),
                            ha='center')
        
        # Save mean curve
        if image_idx is not None:
            mean_curve_path = os.path.join(output_dir, f"latent_means_backward_{image_idx}.png")
        else:
            mean_curve_path = os.path.join(output_dir, "latent_means_backward.png")
        
        plt.savefig(mean_curve_path)
        print(f"Backward mean curve saved to: {mean_curve_path}")
        
        # Plot variance curve
        plt.figure(figsize=(10, 6))
        plt.plot(step_range, latent_vars, marker='o', color='purple')
        plt.title('Backward Variance Change')
        plt.xlabel('Step')
        plt.ylabel('Variance')
        plt.grid(True)
        
        # Add value labels for variance
        for i, var_val in enumerate(latent_vars):
            step_idx = len(latent_vars) - i - 1  # Reverse numbering
            if i % 2 == 0 or var_val > 0.05:  # Every other step or large values
                plt.annotate(f"{var_val:.4f}", 
                            (step_idx, var_val),
                            textcoords="offset points", 
                            xytext=(0,10),
                            ha='center')
        
        # Save variance curve
        if image_idx is not None:
            var_curve_path = os.path.join(output_dir, f"latent_vars_backward_{image_idx}.png")
        else:
            var_curve_path = os.path.join(output_dir, "latent_vars_backward.png")
        
        plt.savefig(var_curve_path)
        print(f"Backward variance curve saved to: {var_curve_path}")
        
        # Combined plot with dual Y-axis
        fig, ax1 = plt.figure(figsize=(12, 7)), plt.gca()
        
        # Mean on primary Y-axis (left)
        ax1.plot(step_range, latent_means, marker='o', color='red', label='Mean')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Mean', color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        
        # Variance on secondary Y-axis (right)
        ax2 = ax1.twinx()
        ax2.plot(step_range, latent_vars, marker='s', color='purple', label='Variance')
        ax2.set_ylabel('Variance', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        
        # Add title and legend
        plt.title('Backward Mean and Variance Change')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Save combined plot
        if image_idx is not None:
            combined_path = os.path.join(output_dir, f"latent_mean_var_backward_combined_{image_idx}.png")
        else:
            combined_path = os.path.join(output_dir, "latent_mean_var_backward_combined.png")
        
        plt.savefig(combined_path, bbox_inches='tight')
        print(f"Combined backward mean and variance plot saved to: {combined_path}")
    
    return latent_means, latent_vars

def plot_combined_mean_curves(forward_means, backward_means, normal_backward_means=None, output_dir=None, start_step=None, image_idx=None):
    """
    将前向和反向扩散过程的潜在表示均值绘制在同一张图上，包括正常反向扩散（无扰动）的曲线
    
    参数:
        forward_means: 前向扩散潜在表示均值列表 (从x0到xN)
        backward_means: 对抗反向扩散潜在表示均值列表 (从xN到x0)
        normal_backward_means: 正常反向扩散潜在表示均值列表 (从xN到x0)，不含对抗扰动
        output_dir: 输出目录
        start_step: 反向扩散的起始步骤(从前向过程中)，例如15表示从x15开始反向扩散
        image_idx: 图像索引，用于保存图表
    """
    # 检查输入
    if not forward_means or not backward_means:
        print("错误: 前向或反向均值数据为空")
        return
    
    # 确保输出目录存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 确定反向扩散的步骤数
    reverse_steps = len(backward_means)
    
    # 准备反向扩散数据的x轴坐标 - 这里是从大到小的反向步骤
    reverse_x = list(range(reverse_steps - 1, -1, -1))  # 反向步骤 (从大到小)
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制对抗反向扩散均值 (红色)
    plt.plot(reverse_x, backward_means, marker='o', color='red', label='adversarial_backward_mean')
    
    # 如果提供了正常反向扩散数据，绘制正常反向扩散曲线 (绿色)
    if normal_backward_means is not None:
        # 确保长度匹配
        if len(normal_backward_means) > len(reverse_x):
            normal_data = normal_backward_means[:len(reverse_x)]
        else:
            normal_data = normal_backward_means
            
        plt.plot(reverse_x[:len(normal_data)], normal_data, marker='s', color='green', label='normal_backward_mean')
        
        # 为正常反向扩散添加标签
        for i, mean_val in enumerate(normal_data):
            if i % 4 == 0:  # 每隔4步显示一个标签，避免过于拥挤
                if i < len(reverse_x):
                    plt.annotate(f"{mean_val:.4f}", 
                                (reverse_x[i], mean_val),
                                textcoords="offset points", 
                                xytext=(0, 25),  # 将标签放在点的上方
                                ha='center',
                                color='green',
                                fontsize=8)
    
    # 处理前向扩散数据
    if start_step is not None and start_step < len(forward_means):
        # 前向扩散数据中，我们要找到对应反向扩散起点的数据
        forward_relevant = forward_means[:start_step+1]  # +1 确保包含第start_step步
        
        # 反转前向数据以匹配反向扩散的步骤顺序（反向步骤从大到小）
        selected_forward_means = forward_relevant[::-1]
        
        # 确保长度不超过反向步骤数
        if len(selected_forward_means) > reverse_steps:
            selected_forward_means = selected_forward_means[:reverse_steps]
        
        # 生成前向数据的x轴坐标
        forward_x = reverse_x[:len(selected_forward_means)]
        
        # 绘制前向扩散均值 (蓝色)
        plt.plot(forward_x, selected_forward_means, marker='x', color='blue', label='forward_mean')
        
        # 添加前向数据的标签
        for i, mean_val in enumerate(selected_forward_means):
            if i % 2 == 0:  # 每隔2步显示
                if i < len(forward_x):
                    plt.annotate(f"{mean_val:.4f}", 
                                (forward_x[i], mean_val),
                                textcoords="offset points", 
                                xytext=(0, -15),
                                ha='center',
                                color='blue',
                                fontsize=8)
    
    # 添加图表元素
    plt.title(f'{start_step if start_step is not None else "N"}start)')
    plt.xlabel('step')
    plt.ylabel('mean')
    plt.grid(True)
    plt.legend()
    
    # 添加每个点的数值标签
    for i, mean_val in enumerate(backward_means):
        if i % 3 == 0 or abs(mean_val) > 0.1:  # 每隔3步或较大值显示
            plt.annotate(f"{mean_val:.4f}", 
                        (reverse_x[i], mean_val),
                        textcoords="offset points", 
                        xytext=(0, 10),
                        ha='center',
                        color='red',
                        fontsize=8)
    
    # 保存图表
    if output_dir:
        if image_idx is not None:
            # 将image_idx转换为字符串
            idx_str = str(image_idx)
            curve_path = os.path.join(output_dir, f"mean_comparison_triple_{idx_str}.png")
        else:
            curve_path = os.path.join(output_dir, "mean_comparison_triple.png")
        
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        print(f"三曲线对比图已保存至: {curve_path}")

def plot_combined_variance_curves(forward_vars, backward_vars, normal_backward_vars=None, output_dir=None, start_step=None, image_idx=None):
    """
    Plot variance curves from forward and backward diffusion on the same graph
    
    Parameters:
        forward_vars: List of variances from forward diffusion (x0 to xN)
        backward_vars: List of variances from adversarial backward diffusion (xN to x0)
        normal_backward_vars: List of variances from normal backward diffusion without perturbation
        output_dir: Output directory
        start_step: Starting step for backward diffusion (from forward process)
        image_idx: Image index for saving the plot
    """
    # Check inputs
    if not forward_vars or not backward_vars:
        print("Error: Forward or backward variance data is empty")
        return
    
    # Ensure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of steps in backward diffusion
    reverse_steps = len(backward_vars)
    
    # Prepare x-axis for backward diffusion data - steps from high to low
    reverse_x = list(range(reverse_steps - 1, -1, -1))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot adversarial backward diffusion variance (red)
    plt.plot(reverse_x, backward_vars, marker='o', color='red', label='adversarial_backward_var')
    
    # If normal backward diffusion data is provided, plot it (green)
    if normal_backward_vars is not None:
        # Ensure length matches
        if len(normal_backward_vars) > len(reverse_x):
            normal_data = normal_backward_vars[:len(reverse_x)]
        else:
            normal_data = normal_backward_vars
            
        plt.plot(reverse_x[:len(normal_data)], normal_data, marker='s', color='green', label='normal_backward_var')
        
        # Add labels for normal backward diffusion
        for i, var_val in enumerate(normal_data):
            if i % 4 == 0:  # Every 4 steps to avoid crowding
                if i < len(reverse_x):
                    plt.annotate(f"{var_val:.4f}", 
                                (reverse_x[i], var_val),
                                textcoords="offset points", 
                                xytext=(0, 25),
                                ha='center',
                                color='green',
                                fontsize=8)
    
    # Process forward diffusion data
    if start_step is not None and start_step < len(forward_vars):
        # Extract relevant forward data up to start_step
        forward_relevant = forward_vars[:start_step+1]  # +1 to include start_step
        
        # Reverse forward data to match backward diffusion step order
        selected_forward_vars = forward_relevant[::-1]
        
        # Ensure length doesn't exceed backward steps
        if len(selected_forward_vars) > reverse_steps:
            selected_forward_vars = selected_forward_vars[:reverse_steps]
        
        # Generate x-axis coordinates for forward data
        forward_x = reverse_x[:len(selected_forward_vars)]
        
        # Plot forward diffusion variance (blue)
        plt.plot(forward_x, selected_forward_vars, marker='x', color='blue', label='forward_var')
        
        # Add labels for forward data
        for i, var_val in enumerate(selected_forward_vars):
            if i % 2 == 0:  # Every 2 steps
                if i < len(forward_x):
                    plt.annotate(f"{var_val:.4f}", 
                                (forward_x[i], var_val),
                                textcoords="offset points", 
                                xytext=(0, -15),
                                ha='center',
                                color='blue',
                                fontsize=8)
    
    # Add chart elements
    plt.title(f'Variance Comparison (Starting from step {start_step if start_step is not None else "N"})')
    plt.xlabel('Step')
    plt.ylabel('Variance')
    plt.grid(True)
    plt.legend()
    
    # Add value labels for adversarial backward diffusion
    for i, var_val in enumerate(backward_vars):
        if i % 3 == 0 or var_val > 0.05:  # Every 3 steps or large values
            plt.annotate(f"{var_val:.4f}", 
                        (reverse_x[i], var_val),
                        textcoords="offset points", 
                        xytext=(0, 10),
                        ha='center',
                        color='red',
                        fontsize=8)
    
    # Save chart
    if output_dir:
        if image_idx is not None:
            # Convert image_idx to string
            idx_str = str(image_idx)
            curve_path = os.path.join(output_dir, f"variance_comparison_triple_{idx_str}.png")
        else:
            curve_path = os.path.join(output_dir, "variance_comparison_triple.png")
        
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        print(f"Triple variance comparison plot saved to: {curve_path}")