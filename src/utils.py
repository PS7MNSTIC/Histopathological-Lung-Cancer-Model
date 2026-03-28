import torch
import os
from src.config import Config

def save_model(model):
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    path = os.path.join(Config.MODEL_DIR, "best_model.pth")
    torch.save(model.state_dict(), path)

def load_model(model):
    path = os.path.join(Config.MODEL_DIR, "best_model.pth")
    model.load_state_dict(torch.load(path))
    return model