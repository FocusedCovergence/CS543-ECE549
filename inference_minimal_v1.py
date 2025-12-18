import torch
import deepinv
from pathlib import Path
import gc

def clear_cuda_cache(device_id=None):
    if torch.cuda.is_available():

        if device_id is not None:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        else:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()

clear_cuda_cache(device_id=7)
device_id = 7
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_size = 224

checkpoint_path = "/shared/BIOE589/siemens/Jhersin_Code/DDPM/src/bening/weights/model_best.pth"
model = deepinv.models.DiffUNet(
    in_channels=3, out_channels=3, pretrained=Path(checkpoint_path)
).to(device)

beta_start = 1e-4
beta_end = 0.02
timesteps = 1000

betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

model.eval()

n_samples = 4

with torch.no_grad():
    x = torch.randn(n_samples, 1, image_size, image_size).to(device)

    for t in reversed(range(timesteps)):
        t_tensor = torch.ones(n_samples, device=device).long() * t

        predicted_noise = model(x, t_tensor, type_t="timestep")

        alpha = alphas[t]
        alpha_cumprod = alphas_cumprod[t]
        beta = betas[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        x = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
        ) + torch.sqrt(beta) * noise

