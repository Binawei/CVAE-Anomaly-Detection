import torch
import torch.nn as nn
import os
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
import config
from model import CVAE
from dataset import get_spits

# def vae_loss(recon_x, x, mu, logvar):
#     bce = nn.functional.binary_cross_entropy(
#         recon_x, x, reduction='sum'
#     )
#     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return bce + kld

def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    # MSE reconstruction (Gaussian likelihood)
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # Gourdeau D, Duchesne S, Archambault L.On the proper use of structural similarity for the robust evaluation of medical image synthesis models.Med Phys.2022 Apr;49(4):2462 - 2474.
    ssim_loss = (1 - ssim(recon_x, x, data_range=1.0, size_average=False)).sum()
    recon = mse + ssim_loss
    # KL divergence (Kingma & Welling, 2014)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # β-VAE weighting (Higgins et al., ICLR 2017)
    loss = recon + beta * kld
    return loss




def train():
    config.setup_reproducibility()
    print(f"Using Device:  {config.DEVICE}")

    train_set, val_set, _ = get_spits()
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)

    model = CVAE(config.LATENT_DIM).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    best_val_loss = float('inf')

    for epoch in tqdm(range(config.EPOCHS), desc="Training"):
        model.train()
        train_loss = 0

        # KL warm-up
        beta = min(0.01, (epoch + 1) / 20)

        for x in tqdm(train_loader, desc=f"Epoch {epoch + 1} [train]", leave=False):
            x = x.to(config.DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)

            loss = vae_loss(recon, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x in tqdm(val_loader, desc=f"Epoch {epoch + 1} [val]", leave=False):
                x = x.to(config.DEVICE)
                recon, mu, logvar = model(x)
                val_loss += vae_loss(recon, x, mu, logvar, beta).item()

        val_loss /= len(val_loader.dataset)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(f"Epoch {epoch + 1}/{config.EPOCHS} | Train Loss: {train_loss: .4f} | Val Loss: {val_loss: .4f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))

        config.empty_cache()
    print(f"\nTraining Complete: Best Val Loss: {best_val_loss: .4f}")

if __name__ == "__main__":
    train()
