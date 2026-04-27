"""
05_gan_augment.py
=================
DCGAN training on GPU for synthetic histopathology image generation.

NOTE: LC25000 classes are balanced (~3,312 training images each).
      This GAN trains on lung_scc images and saves synthetic samples
      to data/gan_synthetic/ for optional use in ablation studies
      (per the Group Anuska spec Step 5 requirement).

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\05_gan_augment.py

Outputs:
    data/gan_synthetic/lung_scc/  — synthetic .png images
    checkpoints/gan_G_final.pth   — saved Generator weights
    results/plots/gan_samples.png — grid of generated samples
"""

import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import pandas as pd
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG, setup_device, ensure_dirs

# ── GAN hyperparams (override CFG defaults here if needed) ───────────────────
IMG_SIZE    = 64        # DCGAN standard — 64×64 (faster than 224, still good quality)
LATENT_DIM  = 100
EPOCHS      = 100
LR          = 2e-4
BETA1       = 0.5       # Adam beta1 for GAN stability
BATCH_SIZE  = 64
N_SYNTHETIC = 500       # how many synthetic images to save after training
SAVE_EVERY  = 10        # save sample grid every N epochs
TARGET_CLASS = "lung_scc"


# ════════════════════════════════════════════════════════════════════════════
#  1.  Dataset — loads a single class from the processed train split
# ════════════════════════════════════════════════════════════════════════════

class SingleClassDataset(Dataset):
    def __init__(self, csv_path: Path, class_name: str, transform=None):
        df = pd.read_csv(csv_path)
        self.df = df[df["label"] == class_name].reset_index(drop=True)
        self.transform = transform
        print(f"[GAN Dataset] {class_name}: {len(self.df)} training images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["processed_path"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        if self.transform:
            img = self.transform(img)
        return img


def get_gan_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],   # normalise to [-1, 1]
                             [0.5, 0.5, 0.5]),
    ])


# ════════════════════════════════════════════════════════════════════════════
#  2.  Generator (64×64 DCGAN)
# ════════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """
    Input:  (N, latent_dim, 1, 1)
    Output: (N, 3, 64, 64)  — values in [-1, 1]
    """
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            # latent_dim → 512 × 4 × 4
            self._block(latent_dim, 512, 4, 1, 0),
            # 512 → 256 × 8 × 8
            self._block(512, 256, 4, 2, 1),
            # 256 → 128 × 16 × 16
            self._block(256, 128, 4, 2, 1),
            # 128 → 64 × 32 × 32
            self._block(128, 64, 4, 2, 1),
            # 64 → 3 × 64 × 64
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    @staticmethod
    def _block(in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        return self.net(z)


# ════════════════════════════════════════════════════════════════════════════
#  3.  Discriminator (64×64 DCGAN)
# ════════════════════════════════════════════════════════════════════════════

class Discriminator(nn.Module):
    """
    Input:  (N, 3, 64, 64)
    Output: (N, 1)  — raw logit (use BCEWithLogitsLoss)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 3 → 64 × 32 × 32
            self._block(3,   64,  4, 2, 1, bn=False),
            # 64 → 128 × 16 × 16
            self._block(64,  128, 4, 2, 1),
            # 128 → 256 × 8 × 8
            self._block(128, 256, 4, 2, 1),
            # 256 → 512 × 4 × 4
            self._block(256, 512, 4, 2, 1),
            # 512 → 1 × 1 × 1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

    @staticmethod
    def _block(in_c, out_c, k, s, p, bn=True):
        layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════════
#  4.  Weight initialisation (DCGAN paper recommendation)
# ════════════════════════════════════════════════════════════════════════════

def weights_init(m):
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cls:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ════════════════════════════════════════════════════════════════════════════
#  5.  Training loop
# ════════════════════════════════════════════════════════════════════════════

def train_gan(device: torch.device):
    ensure_dirs()

    # Output dirs
    syn_dir  = CFG.PROJECT_ROOT / "data" / "gan_synthetic" / TARGET_CLASS
    syn_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = CFG.PLOTS_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_csv = CFG.SPLITS_DIR / "train_processed.csv"
    dataset   = SingleClassDataset(train_csv, TARGET_CLASS, get_gan_transform())
    loader    = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    print(f"[GAN] Batches per epoch: {len(loader)}")

    # Models
    G = Generator(LATENT_DIM).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    # Fixed noise for consistent sample grids
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    # Label smoothing for discriminator stability
    REAL_LABEL = 0.9
    FAKE_LABEL = 0.0

    print(f"\n[GAN] Starting training — {EPOCHS} epochs on {device}")
    print(f"[GAN] Target class: {TARGET_CLASS}  |  Image size: {IMG_SIZE}×{IMG_SIZE}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        G.train(); D.train()
        d_losses, g_losses = [], []
        t0 = time.time()

        for real_imgs in loader:
            real_imgs = real_imgs.to(device, non_blocking=True)
            bs = real_imgs.size(0)

            real_labels = torch.full((bs,), REAL_LABEL, device=device)
            fake_labels = torch.full((bs,), FAKE_LABEL, device=device)

            # ── Train Discriminator ──────────────────────────────────────
            D.zero_grad()
            # Real
            out_real = D(real_imgs).squeeze()
            loss_D_real = criterion(out_real, real_labels)
            # Fake
            noise    = torch.randn(bs, LATENT_DIM, 1, 1, device=device)
            fake_imgs = G(noise).detach()
            out_fake  = D(fake_imgs).squeeze()
            loss_D_fake = criterion(out_fake, fake_labels)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # ── Train Generator ──────────────────────────────────────────
            G.zero_grad()
            noise     = torch.randn(bs, LATENT_DIM, 1, 1, device=device)
            fake_imgs = G(noise)
            out_fake  = D(fake_imgs).squeeze()
            # Generator wants D to output REAL for its fakes
            loss_G = criterion(out_fake, real_labels)
            loss_G.backward()
            opt_G.step()

            d_losses.append(loss_D.item())
            g_losses.append(loss_G.item())

        avg_D = sum(d_losses) / len(d_losses)
        avg_G = sum(g_losses) / len(g_losses)
        elapsed = time.time() - t0

        print(f"  Epoch [{epoch:3d}/{EPOCHS}]  "
              f"D_loss={avg_D:.4f}  G_loss={avg_G:.4f}  "
              f"time={elapsed:.1f}s")

        # Save sample grid every SAVE_EVERY epochs
        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise)           # (64, 3, 64, 64)  in [-1,1]
                samples = (samples + 1) / 2        # rescale to [0,1]
            grid_path = plot_dir / f"gan_epoch_{epoch:03d}.png"
            save_image(samples, grid_path, nrow=8, normalize=False)
            print(f"  → Saved sample grid: {grid_path.name}")

    # ── Save generator weights ────────────────────────────────────────────
    G_path = CFG.CHECKPOINTS_DIR / "gan_G_final.pth"
    CFG.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), G_path)
    print(f"\n[GAN] Generator saved: {G_path}")

    return G, device, syn_dir


# ════════════════════════════════════════════════════════════════════════════
#  6.  Generate & save synthetic images
# ════════════════════════════════════════════════════════════════════════════

def generate_synthetic(G: nn.Module, device: torch.device, syn_dir: Path, n: int = N_SYNTHETIC):
    print(f"\n[GAN] Generating {n} synthetic images → {syn_dir}")
    G.eval()
    saved = 0
    batch = 64
    with torch.no_grad():
        while saved < n:
            bs    = min(batch, n - saved)
            noise = torch.randn(bs, LATENT_DIM, 1, 1, device=device)
            imgs  = G(noise)          # [-1, 1]
            imgs  = (imgs + 1) / 2    # [0, 1]
            imgs  = (imgs * 255).clamp(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()
            for img_arr in imgs:
                out_path = syn_dir / f"syn_{saved:05d}.png"
                Image.fromarray(img_arr).save(out_path)
                saved += 1
    print(f"[GAN] Saved {saved} synthetic images to {syn_dir}")


# ════════════════════════════════════════════════════════════════════════════
#  7.  Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_device()
    device = torch.device(CFG.DEVICE)

    # Check dataset available
    train_csv = CFG.SPLITS_DIR / "train_processed.csv"
    if not train_csv.exists():
        print(f"[ERROR] Missing: {train_csv}")
        print("  → Run 03_preprocessing.py first.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" STEP 5b: GAN AUGMENTATION (DCGAN on GPU)")
    print("=" * 60)
    print(f" Latent dim : {LATENT_DIM}")
    print(f" Image size : {IMG_SIZE}×{IMG_SIZE}")
    print(f" Epochs     : {EPOCHS}")
    print(f" Batch size : {BATCH_SIZE}")
    print(f" Target     : {TARGET_CLASS}")
    print("=" * 60 + "\n")

    G, device, syn_dir = train_gan(device)
    generate_synthetic(G, device, syn_dir, n=N_SYNTHETIC)

    print("\n[OK] 05_gan_augment.py complete.")
    print(f"     Synthetic images: {syn_dir}")
    print(f"     Generator weights: {CFG.CHECKPOINTS_DIR / 'gan_G_final.pth'}")
    print(f"     Next: python src\06_model_hagcanet.py  (or: python main.py)")

