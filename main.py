import argparse, os
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import time
from ldm.util import instantiate_from_config
from scripts.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import random  
import ldm.modules.attention 

def orthogonal_gradient_projection(adv_grad, denoise_grad):
    """
    Project the adversarial gradient onto the direction orthogonal to the denoising gradient.
    """
    denoise_grad_norm = torch.norm(denoise_grad, p=2)
    if denoise_grad_norm > 0:
        denoise_direction = denoise_grad / denoise_grad_norm
    else:
        return adv_grad
    
    projection = torch.sum(adv_grad * denoise_direction) * denoise_direction
    
    orthogonal_component = adv_grad - projection
    
    return orthogonal_component

def adain(cnt_feat, sty_feat, fixed=False):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)

    sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)

    if fixed:
        sty_mean = torch.zeros_like(cnt_mean)
        sty_std = torch.ones_like(cnt_std)

    output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
    return output

def latent2image_with_grad(model, latents):
    """Decode latents with model.first_stage_model (VAE) while preserving gradients."""
    latents = 1 / 0.18215 * latents
    decoded = model.first_stage_model.decode(latents)
    return decoded

def encode_with_fixed_seed(model, image, seed=8888):
    """Encode the image with a fixed random seed."""
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    with torch.no_grad():
        latents = model.get_first_stage_encoding(model.encode_first_stage(image))
    torch.set_rng_state(rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    return latents

class ImageClassifier:
    def __init__(self, model_name='resnet50', device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Initializing classifier {model_name}...")
        self.model_name = model_name.lower()
        self.input_size = 299 if self.model_name == 'inception_v3' else 224
        self.model = self._load_model()
        self.model.eval()  # Ensure the classifier is in evaluation mode.
        for param in self.model.parameters():
            param.requires_grad_(False)  # Freeze classifier parameters.
        

    def _load_model(self):
        """Load the specified pretrained model."""
        if self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif self.model_name == 'vgg19':
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        elif self.model_name == 'inception_v3':
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
            model.aux_logits = False  # Disable the auxiliary classifier.
        elif self.model_name == 'convnext':
            model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        elif self.model_name == 'vit_b_16':
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif self.model_name == 'swin_b':
            model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model.to(self.device)
        

class BackgroundReconstructor:
    def __init__(self, model):
        self.model = model
        self.alphas_cumprod = model.alphas_cumprod
    
    @torch.no_grad()
    def forward_diffusion(self, x, steps, conditioning, unconditional_conditioning=None, 
                        unconditional_guidance_scale=1.0, order=3, save_steps=False, 
                        output_dir=None, save_interval=5, store_intermediates=False):
        """Run forward diffusion with DPM-Solver and optionally store intermediate states."""
        if save_steps and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        device = self.model.betas.device
        batch_size = x.shape[0]
        
        # Store intermediate states when requested.
        intermediate_states = [] if store_intermediates else None
        
        # Use the same alphas_cumprod schedule as the backward DDIM process.
        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)
        
        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            ns, model_type="noise", guidance_type="classifier-free",
            condition=conditioning, unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
        )
                
        dpm_solver = DPM_Solver(model_fn, ns)
        
        t_0 = 1. / dpm_solver.noise_schedule.total_N
        t_T = dpm_solver.noise_schedule.T
        timesteps = torch.linspace(t_0, t_T, steps + 1).to(device)

        if store_intermediates:
            intermediate_states.append({
                "step": 0, 
                "latent": x.clone(),
                "timestep": timesteps[0]
            })
        
        # Run DPM-Solver.
        vec_t = timesteps[0].expand((batch_size))
        model_prev_list = [dpm_solver.model_fn(x, vec_t)]
        t_prev_list = [vec_t]
        
        # Initialize the first few steps.
        for init_step in range(1, order):
            vec_t = timesteps[init_step].expand(batch_size)
            # print(vec_t)

            # transformed_noise_pred = dpm_solver.model_fn(x, vec_t)
            # f_std = transformed_noise_pred.std(dim=[0, 2, 3], keepdim=True)
            # print(f"Per-channel noise std: f_std={f_std.squeeze().cpu().detach().numpy()}")

            x = dpm_solver.multistep_dpm_solver_update(
                x, model_prev_list, t_prev_list, vec_t, init_step, 
                solver_type='dpmsolver'
            )

            model_prev_list.append(dpm_solver.model_fn(x, vec_t))
            t_prev_list.append(vec_t)
            
            if store_intermediates:
                intermediate_states.append({
                    "step": init_step,
                    "latent": x.clone(),
                    "timestep": timesteps[init_step]
                })
                
            if save_steps and output_dir:
                self._save_intermediate(x, init_step, output_dir, "encode", batch_size)

        for step in range(order, steps + 1):
            vec_t = timesteps[step].expand(batch_size)
            x = dpm_solver.multistep_dpm_solver_update(
                x, model_prev_list, t_prev_list, vec_t, order,
                solver_type='dpmsolver'
            )

            model_prev_list.append(dpm_solver.model_fn(x, vec_t))
            t_prev_list.append(vec_t)
            model_prev_list.pop(0)
            t_prev_list.pop(0)
            
            if store_intermediates:
                intermediate_states.append({
                    "step": step,
                    "latent": x.clone(),
                    "timestep": timesteps[step]
                })
                
                
            if save_steps and output_dir and step % save_interval == 0:
                self._save_intermediate(x, step, output_dir, "encode", batch_size)
        
        if store_intermediates:
            return x, intermediate_states
        return 
    
    # Time-step logic for the backward diffusion process.
    def backward_diffusion(self, x, steps, conditioning, unconditional_conditioning=None, 
        unconditional_guidance_scale=1.0, eta=0.0, save_steps=False,
        output_dir=None, save_interval=5, start_step=None,
        true_label=None, classifier=None,
        apply_adv=False, adv_start_step=12, adv_end_step=15, 
        adv_epsilon=0.01, image_idx=None,
        forward_states=None,
        momentum=0.9  
    ):

        if save_steps and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        device = self.model.betas.device
        batch_size = x.shape[0]
        
        # Set time steps and ensure the type is long.
        timesteps = torch.linspace(
            self.model.num_timesteps - 1, 0, steps + 1
        ).int().to(device)
        
        if start_step is not None:
            start_idx = steps - start_step
            timesteps = timesteps[start_idx:]
            print(f"Starting backward diffusion from step {start_step}; {len(timesteps)-1} steps remaining")

        if save_steps and output_dir:
            self._save_intermediate_with_grad(x, 0, output_dir, "decode", batch_size)

        alphas = self.model.alphas_cumprod

        total_steps = steps
        start_offset = total_steps - len(timesteps)

        # Momentum-related state.
        # momentum = 0.9  # Momentum coefficient.
        prev_grad = None  # Store the gradient from the previous step.

        if forward_states is not None:
            # Reverse forward states so they align with the backward trajectory.
            reversed_forward_states = list(reversed(forward_states))
            print(f"Loaded {len(reversed_forward_states)} forward diffusion states")

        # Run DDIM sampling.
        for i, step in enumerate(timesteps[:-1]):
            clean_x = x
            current_step = i + start_offset
            print(f"\nCurrent step: {current_step}/{total_steps}")  # Use the actual number of total steps.

            ts = step.long().expand(batch_size)
            ts_next = timesteps[i + 1].long().expand(batch_size)

            start_a = alphas[timesteps[1]]
            a_t = alphas[ts].view(-1, 1, 1, 1)
            a_prev = alphas[ts_next].view(-1, 1, 1, 1)

            cnt_mean = x.mean(dim=[0, 2, 3], keepdim=True)
            cnt_std = x.std(dim=[0, 2, 3], keepdim=True)
            # print(f"Current step mean and std: cnt_mean={cnt_mean.squeeze().cpu().detach().numpy()}, cnt_std={cnt_std.squeeze().cpu().detach().numpy()}")

            # Apply perturbations within the adversarial attack interval.
            if apply_adv and adv_start_step <= current_step <= adv_end_step:
                # Compute epsilon for the current step.
                steps_in_attack = current_step - adv_start_step
                current_epsilon = adv_epsilon 
                
                x = x.detach().requires_grad_(True)

                
                with torch.enable_grad():
                    recon_latents = x.clone()

                    remaining_steps = len(timesteps) - i -1

                    adap = torch.sqrt(1 - a_prev).item()/torch.sqrt(1 - start_a).item()
                    current_epsilon = adap * current_epsilon
                    
                    # print("Reconstruction process ********************")
                    for step_idx in range(remaining_steps):               
                        target_idx = current_step + step_idx + 1
                        # print(target_idx)
                        forward_x = reversed_forward_states[target_idx]["latent"]
                        recon_latents = adain(recon_latents,forward_x)

                        curr_t = timesteps[i + step_idx].long().expand(batch_size)
                        next_t = timesteps[i + step_idx + 1].long().expand(batch_size)
                        
                        noise_pred = self.model.apply_model(
                            recon_latents, curr_t, conditioning
                        )

                        # DDIM update.
                        curr_alpha = alphas[curr_t].view(-1, 1, 1, 1)
                        curr_alpha_prev = alphas[next_t].view(-1, 1, 1, 1) 
                        pred_x0 = (recon_latents - torch.sqrt(1. - curr_alpha) * noise_pred) / torch.sqrt(curr_alpha)
                        sigma = eta * torch.sqrt((1 - curr_alpha_prev) / (1 - curr_alpha)) * torch.sqrt(1 - curr_alpha / curr_alpha_prev)
                        recon_latents = torch.sqrt(curr_alpha_prev) * pred_x0 + torch.sqrt(1 - curr_alpha_prev - sigma**2) * noise_pred
                        if target_idx == 29:
                            last_idx = target_idx + 1
                            forward_x = reversed_forward_states[last_idx]["latent"]
                            recon_latents = adain(recon_latents,forward_x)

                        del noise_pred
                        torch.cuda.empty_cache()

                    # print("Reconstruction finished ********************")
                        
                    
                    # Compute the final prediction and loss.
                    decoded = latent2image_with_grad(self.model, recon_latents)
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)

                    resized = F.interpolate(decoded, 
                                        size=(classifier.input_size, classifier.input_size),
                                        mode='bilinear', 
                                        align_corners=False)
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                    normalized = (resized - mean) / std
                    
                    del decoded, resized
                                        
                    outputs = classifier.model(normalized)
                    loss = F.cross_entropy(outputs, torch.tensor([true_label]).to(device))
                    
                    with torch.no_grad():
                        probs = F.softmax(outputs, dim=1)
                        pred_prob, pred_class = torch.max(probs, 1)
                        print(f"\nPrediction result at attack step {current_step}:")
                        print(f"Predicted class: {pred_class.item()}, true label: {true_label}")
                        print(f"Confidence: {pred_prob.item():.4f}, loss: {loss.item():.4f}")

                    with torch.no_grad():
                        target_idx = current_step + 1
                        print(target_idx)
                        # forward_x = reversed_forward_states[target_idx]["latent"]
                        # clean_x = adain(clean_x,forward_x)
                        # # Get the denoising gradient (predicted noise).
                        # ts = step.long().expand(batch_size)
                        # denoise_pred = self.model.apply_model(clean_x, ts, conditioning)
                        # # The denoising gradient follows the predicted-noise direction.
                        # denoise_grad = denoise_pred

                    loss.backward()
                    current_grad = x.grad
                    if prev_grad is None:
                        accumulated_grad = current_grad
                    else:
                        accumulated_grad = momentum * prev_grad + current_grad

                    prev_grad = accumulated_grad.clone().detach()           
                    # Use the momentum-accumulated gradient.
                    orthogonal_grad = accumulated_grad

                    grad_norm = torch.norm(orthogonal_grad)
                    if grad_norm > 0:
                        orthogonal_grad = orthogonal_grad / grad_norm

                    # orthogonal_grad = orthogonal_gradient_projection(current_epsilon * orthogonal_grad.sign(), denoise_grad)           

                    print(f"\nCurrent epsilon: {current_epsilon:.4f}")
                    x = x + current_epsilon * orthogonal_grad.sign()
                    # x = x + orthogonal_grad
                    x = x.detach()

            target_idx = current_step + 1
            print(target_idx)
            forward_x = reversed_forward_states[target_idx]["latent"]
            x = adain(x,forward_x)
                        
            with torch.no_grad():
                e_adv = self.model.apply_model(
                    x, ts, conditioning,
                )
                e_t = e_adv 
                pred_x0 = (x - torch.sqrt(1. - a_t) * e_t) / torch.sqrt(a_t)

                sigma_t = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(1 - a_t / a_prev)
                noise = torch.randn_like(x) if eta > 0 else 0

                x = torch.sqrt(a_prev) * pred_x0 + \
                    torch.sqrt(1 - a_prev - sigma_t**2) * e_t + \
                    sigma_t * noise

                if current_step == 28:
                    target_idx = current_step + 2
                    print(target_idx)
                    forward_x = reversed_forward_states[target_idx]["latent"]
                    x = adain(x,forward_x)
                
                print(torch.sqrt(1 - a_prev - sigma_t**2).item())
                del pred_x0, e_t

            if save_steps and output_dir and i % save_interval == 0:
                self._save_intermediate_with_grad(x, i+1, output_dir, "decode", batch_size)
            
            torch.cuda.empty_cache()
        
        return x
    
    def _save_intermediate(self, x, step, output_dir, prefix, batch_size):
        """Save an intermediate step with no_grad."""
        with torch.no_grad():
            self._save_intermediate_with_grad(x, step, output_dir, prefix, batch_size)
    
    def _save_intermediate_with_grad(self, x, step, output_dir, prefix, batch_size):
        """Save an intermediate step while preserving gradient flow."""
        try:
            decoded = latent2image_with_grad(self.model, x)
            decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
            
            for i in range(min(batch_size, 4)):  # Save at most 4 samples.
                img = 255. * rearrange(decoded[i].detach().cpu().numpy(), 'c h w -> h w c')
                Image.fromarray(img.astype(np.uint8)).save(
                    os.path.join(output_dir, f"{prefix}_step_{step}_sample_{i}.png")
                )
        except Exception as e:
            print(f"Failed to save intermediate state: {e}")


def load_model_from_config(config, ckpt, device):
    print(f"Loading model: {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def load_img(path, size=512):
    """Load and preprocess an image."""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"Loading input image ({w}x{h}) from: {path}")
    
    image = image.resize((size, size), resample=PIL.Image.LANCZOS)
    
    # Convert to a PyTorch tensor and normalize to [-1, 1].
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return torch.from_numpy(image) * 2.0 - 1.0

def main():
    parser = argparse.ArgumentParser(description="Gradient-aware background image reconstruction")
    parser.add_argument("--input_dir", type=str, default="images", help="Input image directory")
    parser.add_argument("--label_file", type=str, default="labels.txt", help="Path to the label file")
    parser.add_argument("--output_dir", type=str, default="output_adv", help="Output directory")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt")
    parser.add_argument("--dpm_steps", type=int, default=30, help="Number of DPM-Solver steps")
    parser.add_argument("--ddim_steps", type=int, default=30, help="Number of DDIM steps") 
    parser.add_argument("--dpm_order", type=int, default=2, choices=[1, 2, 3], help="DPM-Solver order")
    parser.add_argument("--ddim_eta", type=float, default=0, help="DDIM stochasticity parameter")
    parser.add_argument("--scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--config", type=str, default="./configs/stable-diffusion/v2-inference.yaml", help="Model config")
    parser.add_argument("--ckpt", type=str, default="./ckpt/512-base-ema.ckpt", help="Model checkpoint")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--gpu", type=str, default="cuda:0", help="GPU device")
    parser.add_argument("--save_steps", action="store_true", help="Whether to save intermediate steps")
    parser.add_argument("--save_interval", type=int, default=1, help="Interval for saving intermediate steps")
    parser.add_argument("--img_size", type=int, default=384, help="Image size")
    parser.add_argument("--start_step", type=int, default=None, help="Start reconstruction from a specified DPM forward step (0-based)")
    parser.add_argument("--enable_grad", action="store_true", help="Enable gradient propagation for debugging")
    parser.add_argument("--apply_adv", action="store_true", help="Whether to enable adversarial attack")
    parser.add_argument("--adv_start", type=int, default=12, help="Adversarial perturbation start step")
    parser.add_argument("--adv_end", type=int, default=15, help="Adversarial perturbation end step")
    parser.add_argument("--adv_epsilon", type=float, default=0.01, help="Adversarial perturbation magnitude")
    parser.add_argument("--target_model", type=str, default="resnet50",
                        choices=['resnet50', 'vgg19', 'mobilenet_v2', 'inception_v3', 
                                'convnext', 'vit_b_16', 'swin_b'],
                        help="Target classifier model")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum coefficient for adversarial gradient accumulation")
    
    opt = parser.parse_args()
    device = torch.device(opt.gpu) if torch.cuda.is_available() else torch.device("cpu")

    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.makedirs(os.path.join(opt.output_dir, "adv"), exist_ok=True)

    with open(opt.label_file, "r") as f:
        labels = [int(x.strip()) - 1 for x in f.readlines()]  # Convert 1-indexed labels to 0-indexed labels.

    csv_path = os.path.join(opt.output_dir, "attack_results.csv")
    with open(csv_path, "w") as f:
        f.write("image_idx,true_label,initial_prediction,initial_confidence,final_prediction,final_confidence,attack_success,start_step\n")    

    config = OmegaConf.load(opt.config)
    # Original text encoder setup:
    # model = load_model_from_config(config, opt.ckpt, device)
    #
    # HAM disables the original text cross-attention path, so the OpenCLIP text
    # encoder is not needed for the default attack. Marking the condition stage
    # as unconditional avoids downloading the 3.94GB OpenCLIP checkpoint.
    config.model.params.cond_stage_config = "__is_unconditional__"
    config.model.params.force_null_conditioning = True
    model = load_model_from_config(config, opt.ckpt, device)
    classifier = ImageClassifier(model_name=opt.target_model, device=device) if opt.apply_adv else None

    if opt.enable_grad:
        model.first_stage_model.requires_grad_(True)
    
    with torch.no_grad():
        # Original text conditioning:
        # c = model.get_learned_conditioning([opt.prompt])
        # uc = model.get_learned_conditioning([""])
        # cond = c
        # uncond = uc
        #
        # Keep the cross-attention context shape expected by SD2 without using
        # text semantics. Shape: batch=1, tokens=77, context_dim=1024.
        cond = torch.zeros((1, 77, 1024), device=device)
        uncond = cond

    reconstructor = BackgroundReconstructor(model)

    for idx in range(1, 1001):  # 1 to 1000.
        image_path = os.path.join(opt.input_dir, f"{idx}.png")
        true_label = labels[idx-1]
        print(f"\nProcessing image {idx}/1000: {image_path}")
        print(f"True label: {true_label}")

        init_image = load_img(image_path, opt.img_size).to(device)

        with torch.no_grad():
            initial_img = encode_with_fixed_seed(model, init_image, seed=8888)
            initial_decoded = latent2image_with_grad(model, initial_img)
            initial_decoded = (initial_decoded / 2 + 0.5).clamp(0, 1)
            resized = F.interpolate(initial_decoded, 
                                size=(classifier.input_size, classifier.input_size), 
                                mode='bilinear', 
                                align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            normalized = (resized - mean) / std
            outputs = classifier.model(normalized)
            probs = F.softmax(outputs, dim=1)
            initial_conf, initial_pred = torch.max(probs, 1)
        
        with torch.no_grad():
            init_latent = encode_with_fixed_seed(model, init_image, seed=8888)
        
        print(f"=== Starting reconstruction for image {idx} ===")
        start_time = time.time()
        
        # Forward diffusion process.
        print(f"[1/2] Forward diffusion (DPM-Solver, {opt.dpm_steps} steps, order={opt.dpm_order})...")
        ctx = nullcontext() if opt.enable_grad else torch.no_grad()
        with ctx, autocast("cuda"):
            noisy_latent, intermediate_states = reconstructor.forward_diffusion(
                init_latent, opt.dpm_steps, cond, uncond, opt.scale,
                opt.dpm_order, False, None, opt.save_interval,
                store_intermediates=True
            )

        for state in intermediate_states:
            step = state["step"]
            latent = state["latent"]
            # Compute the mean and standard deviation.
            mean_val = latent.mean(dim=[0, 2, 3], keepdim=True)
            std_val = latent.std(dim=[0, 2, 3], keepdim=True)
            # print(f"{step} mean and std: cnt_mean={mean_val.squeeze().cpu().detach().numpy()}, cnt_std={std_val.squeeze().cpu().detach().numpy()}")

        
        # Determine the starting state.
        if opt.start_step is not None and 0 <= opt.start_step < len(intermediate_states):
            start_state = intermediate_states[opt.start_step]
            start_latent = start_state["latent"]
            print(f"Starting reconstruction from fixed step {opt.start_step}/{opt.dpm_steps}...")
        else:
            start_latent = noisy_latent
        
        # Backward diffusion process.
        print(f"[2/2] Backward diffusion (DDIM, {opt.ddim_steps} steps, eta={opt.ddim_eta})...")
        ctx = nullcontext() if (opt.enable_grad or opt.apply_adv) else torch.no_grad()
        
        with torch.set_grad_enabled(True), ctx, autocast("cuda"):
            # Call backward_diffusion with the current image index.
            reconstructed = reconstructor.backward_diffusion(
                start_latent, opt.ddim_steps, cond, uncond, opt.scale,
                opt.ddim_eta, opt.save_steps, opt.output_dir, opt.save_interval,
                start_step=opt.start_step,
                true_label=true_label,
                classifier=classifier,
                apply_adv=opt.apply_adv,
                adv_start_step=opt.adv_start,
                adv_end_step=opt.adv_end,
                adv_epsilon=opt.adv_epsilon,
                image_idx=idx,
                forward_states=intermediate_states,
                momentum=opt.momentum       
            )
        
        with torch.no_grad():
            reconstructed_latents = reconstructed  # Latent representation returned by backward_diffusion.

            # Decode the final adversarial latent.
            final_img_with_residual = model.decode_first_stage(reconstructed_latents)
            
            # Postprocess for visualization and evaluation.
            final_img_with_residual = (final_img_with_residual / 2 + 0.5).clamp(0, 1)
            
            img_with_residual = 255. * rearrange(final_img_with_residual[0].cpu().numpy(), 'c h w -> h w c')
            adv_path = os.path.join(opt.output_dir, "adv", f"{idx}.png")
            Image.fromarray(img_with_residual.astype(np.uint8)).save(adv_path)

            elapsed = time.time() - start_time
            print(f"=== Image {idx} completed in {elapsed:.2f}s ===")
            
            # Evaluate the final adversarial sample with the classifier.
            resized = F.interpolate(final_img_with_residual, size=(224, 224), mode='bilinear', align_corners=False)
            normalized = (resized - mean) / std
            
            outputs = classifier.model(normalized)
            probs = F.softmax(outputs, dim=1)
            final_conf, final_pred = torch.max(probs, 1)
            
            print(f"\n=== Attack result for image {idx} ===")
            print(f"Original label: {true_label}")
            print(f"Initial prediction: {initial_pred.item()} (confidence: {initial_conf.item():.4f})")
            print(f"Final prediction: {final_pred.item()} (confidence: {final_conf.item():.4f})")
            
            with open(csv_path, "a") as f:
                f.write(f"{idx},{true_label},{initial_pred.item()},{initial_conf.item():.4f},"
                        f"{final_pred.item()},{final_conf.item():.4f},"
                        f"{final_pred.item() != true_label},{opt.start_step}\n")
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
