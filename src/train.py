import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from src.config import Config
from src.utils import save_model
from tqdm import tqdm

def train(model, train_loader, val_loader):
    device = Config.DEVICE
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    scaler = GradScaler(enabled=Config.USE_AMP)

    best_loss = float("inf")
    patience_counter = 0


    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")

        for x, y in loop:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            with autocast(enabled=Config.USE_AMP):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        scheduler.step()

        val_loss = validate(model, val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered")
            break


def validate(model, loader):
    model.eval()
    loss_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            out = model(x)
            loss_total += criterion(out, y).item()

    return loss_total / len(loader)

from sklearn.model_selection import GroupKFold

def train_kfold(model, image_paths, labels, groups):
    gkf = GroupKFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(image_paths, labels, groups)):
        print(f"Fold {fold+1}")

        train_dataset = LungDataset(
            [image_paths[i] for i in train_idx],
            [labels[i] for i in train_idx]
        )

        val_dataset = LungDataset(
            [image_paths[i] for i in val_idx],
            [labels[i] for i in val_idx]
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        train(model, train_loader, val_loader)