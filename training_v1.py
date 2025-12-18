# %%
from deepinv.models.diffunet import DiffUNet
import os

import matplotlib.pyplot as plt

import torch
import deepinv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
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

class SimpleFitzDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = ['Benign', 'Malignant']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = "/shared/BIOE589/siemens/Jhersin_Code/CS543-ECE549/Classification/data/" + row["md5hash"] + ".jpg"
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        label = int(row["malignant"])# 1 for malignant, 0 for benign
        return img, label


# class SimpleFitzDataset(Dataset):
#     def __init__(self, df, transform=None):
#         self.df = df.reset_index(drop=True)
#         self.transform = transform
#         # Base image directory - ADJUST THIS!
#         self.base_path = "/shared/BIOE589/siemens/Jhersin_Code/CS543-ECE549/Classification/data"
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         # Convert idx to integer if it's a tensor or numpy array
#         if isinstance(idx, (torch.Tensor, np.integer)):
#             idx = int(idx)
#         elif isinstance(idx, np.ndarray):
#             idx = int(idx.item())
#
#         row = self.df.iloc[idx]
#
#         # Construct image path
#         img_path = f"{self.base_path}/{row['md5hash']}.jpg"
#
#         # Load image
#         img = Image.open(img_path).convert("RGB")
#
#         if self.transform:
#             img = self.transform(img)
#
#         # Get label
#         label = int(row["malignant"])  # 1 for malignant, 0 for benign
#
#         return img, label

def main():
    # Set the device
    clear_cuda_cache(device_id=7)
    device_id = 7
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set hyperparameters
    batch_size = 4
    num_epochs = 200
    lr = 1e-4
    image_size = 224

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )

    # Creation of dataset
    # csv_path = '/shared/BIOE589/siemens/Jhersin_Code/CS543-ECE549/Classification/csv/fitzpatrick17k_downloaded.csv'
    # df = pd.read_csv(csv_path)
    # benign_df = df[df['three_partition_label'] == 'benign'].copy()
    # train_dataset = SimpleFitzDataset(df, transform=transform)

    fitzpatrick_df = pd.read_csv(
        "/shared/BIOE589/siemens/Jhersin_Code/CS543-ECE549/Classification/csv/fitzpatrick17k_downloaded.csv")
    fitzpatrick_df = fitzpatrick_df[(fitzpatrick_df["fitzpatrick_scale"] >= 0)].copy()
    fitzpatrick_df["malignant"] = (fitzpatrick_df["three_partition_label"] == "malignant").astype(int)
    skin_tone_map = {
        1: 12, 2: 12,
        3: 34, 4: 34,
        5: 56, 6: 56,
    }
    fitzpatrick_df["skin_tone"] = fitzpatrick_df["fitzpatrick_scale"].map(skin_tone_map).astype(int)

    train_dataset = SimpleFitzDataset(fitzpatrick_df, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # ===== MODEL SETUP =====
    model = DiffUNet(
        in_channels=3, out_channels=3, pretrained="download", large_model=False
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = deepinv.loss.MSE()

    # DDPM constants
    timesteps = 1000
    beta_start = 1e-4
    beta_end = 0.02

    # Noise schedule
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Track best loss for model saving
    best_loss = float('inf')

    # Create directories
    os.makedirs("./Total/losses", exist_ok=True)
    os.makedirs("./Total/weights", exist_ok=True)
    os.makedirs("./Total/img", exist_ok=True)

    # Training loop
    all_losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_losses = []

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()

            # Sample timesteps and noise
            t = torch.randint(0, timesteps, (images.shape[0],), device=device)
            # Create noise
            noise = torch.randn_like(images)

            # Apply forward diffusion
            noised_images = (
                    sqrt_alphas_cumprod[t, None, None, None] * images
                    + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
            )

            # Predict noise
            noise_pred = model(noised_images, t, type_t="timestep")[:, :3, :, :] # mean and variance

            # Calculate loss
            loss = mse(noise_pred, noise).mean()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value
            epoch_losses.append(loss_value)

        all_losses.extend(epoch_losses)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "./Total/weights/model_best.pth")
            print(f"  Saved best model with loss: {best_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"./Total/weights/model_epoch_{epoch + 1}.pth")

        # Visualization every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                # Get a sample image
                sample_image, _ = train_dataset[0]
                sample_image = sample_image.unsqueeze(0).to(device)

                # Choose a fixed timestep for consistent comparison
                t_val = 500
                t_tensor = torch.tensor([t_val], device=device)

                # Create noise and noisy image
                noise = torch.randn_like(sample_image)
                noised_image = (
                        sqrt_alphas_cumprod[t_val] * sample_image
                        + sqrt_one_minus_alphas_cumprod[t_val] * noise
                )

                # Denoise
                noise_pred = model(noised_image, t_tensor, type_t="timestep")[:, :3, :, :]
                denoised_image = (
                                         noised_image - sqrt_one_minus_alphas_cumprod[t_val] * noise_pred
                                 ) / sqrt_alphas_cumprod[t_val]
                denoised_image = torch.clamp(denoised_image, -1, 1)

                # Create side-by-side comparison
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))

                # Original
                axes[0].imshow(sample_image.cpu().squeeze(0).permute(1, 2, 0))
                axes[0].set_title("Original")
                axes[0].axis("off")

                # Noisy
                axes[1].imshow(noised_image.cpu().squeeze(0).permute(1, 2, 0))
                axes[1].set_title(f"Noisy (t={t_val})")
                axes[1].axis("off")

                # Denoised
                axes[2].imshow(denoised_image.cpu().squeeze(0).permute(1, 2, 0))
                axes[2].set_title("Denoised")
                axes[2].axis("off")

                plt.tight_layout()
                plt.savefig(f"./Total/img/epoch_{epoch + 1}_comparison.png",
                            bbox_inches='tight', dpi=150)
                plt.close()

    # Save final model
    torch.save(model.state_dict(), "./Total/weights/model_final.pth")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(all_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./bening_1/losses/loss_curve.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()