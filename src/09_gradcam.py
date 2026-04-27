"""
09_gradcam.py
=============
Generates Grad-CAM heatmap overlays for HAGCA-Net on a sample of test images.

How Grad-CAM works here:
  - Hook is attached to the last conv layer of EfficientNet-B3 (CNN branch)
  - Forward pass computes activations; backward pass from target class scores
    computes gradients w.r.t. that layer's feature maps
  - Global average-pooled gradients weight the feature maps -> heat map
  - Heat map is resized to 224x224 and overlaid on the original image

Outputs (per class, 5 images each):
  results/gradcam/<class_name>/img_<n>_gradcam.png

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\09_gradcam.py
"""

import sys, importlib.util, random
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch.nn.functional import interpolate

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))
from config import CFG, setup_device, ensure_dirs, get_logger

def _load(alias, fname):
    spec = importlib.util.spec_from_file_location(alias, SRC / fname)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_aug   = _load("aug",   "04_augmentation.py")
_model = _load("model", "06_model_hagcanet.py")

LungDataset        = _aug.LungDataset
get_val_transforms = _aug.get_val_transforms
HAGCANet           = _model.HAGCANet


# ════════════════════════════════════════════════════════════════════════════
#  Grad-CAM hook manager
# ════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Grad-CAM for an arbitrary conv layer.

    Parameters
    ----------
    model   : nn.Module  — full HAGCA-Net (must be in eval mode)
    layer   : nn.Module  — the target convolutional layer to hook
    """

    def __init__(self, model, layer):
        self.model       = model
        self.activations = None
        self.gradients   = None

        # Forward hook — capture layer output
        self._fwd_hook = layer.register_forward_hook(self._save_activation)
        # Backward hook — capture gradients flowing INTO the layer output
        self._bwd_hook = layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()          # (B, C, H, W)

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()    # (B, C, H, W)

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    @torch.enable_grad()
    def generate(self, img_tensor, target_class=None):
        """
        Parameters
        ----------
        img_tensor   : (1, 3, 224, 224) tensor on device
        target_class : int or None (None -> argmax of model output)

        Returns
        -------
        cam  : (224, 224) numpy array, values in [0, 1]
        pred : int — predicted class index
        conf : float — softmax confidence of the predicted class
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward
        logits = self.model(img_tensor)                       # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)
        pred   = int(logits.argmax(1).item())
        conf   = float(probs[0, pred].item())

        tgt = target_class if target_class is not None else pred

        # Backward from target class score (not softmax — raw logit)
        score = logits[0, tgt]
        score.backward()

        # Grad-CAM weights = global-average-pooled gradients over spatial dims
        # gradients: (1, C, H, W) -> weights: (C,)
        weights = self.gradients[0].mean(dim=(1, 2))          # (C,)

        # Weighted combination of activation maps
        acts = self.activations[0]                            # (C, H, W)
        cam  = (weights[:, None, None] * acts).sum(0)         # (H, W)
        cam  = torch.relu(cam)                                # keep positives only

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = torch.zeros_like(cam)

        # Upsample to image size
        cam_up = interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()                              # (224, 224)

        return cam_up, pred, conf


# ════════════════════════════════════════════════════════════════════════════
#  Overlay helper
# ════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def tensor_to_rgb(t):
    """Convert a normalised (3,224,224) tensor back to uint8 (224,224,3)."""
    img = t.cpu().numpy().transpose(1, 2, 0)          # (H,W,3)
    img = img * IMAGENET_STD + IMAGENET_MEAN           # undo normalise
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def overlay_cam(img_rgb, cam, alpha=0.45):
    """Blend Grad-CAM heatmap with the original RGB image."""
    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )                                                  # (H,W,3) BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap + (1 - alpha) * img_rgb).astype(np.uint8)
    return blended


def save_gradcam_figure(img_rgb, cam, blended, true_cls, pred_cls,
                        conf, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(blended)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")

    status = "CORRECT" if true_cls == pred_cls else "WRONG"
    color  = "green"   if true_cls == pred_cls else "red"
    fig.suptitle(
        f"True: {true_cls}  |  Pred: {pred_cls}  ({conf*100:.1f}%)  [{status}]",
        fontsize=12, color=color, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

SAMPLES_PER_CLASS = 5          # images to visualise per class
RANDOM_SEED       = 42


def main():
    setup_device()
    ensure_dirs()
    device = torch.device(CFG.DEVICE)
    logger = get_logger("gradcam")

    logger.info("=" * 60)
    logger.info("  STEP 9: GRAD-CAM EXPLAINABILITY")
    logger.info("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────
    ckpt_path = CFG.BEST_MODEL_PATH
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = HAGCANet(num_classes=CFG.NUM_CLASSES, pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info(f"Loaded: {ckpt_path.name}  (epoch {ckpt.get('epoch','?')})")

    # ── Hook target layer: last conv in EfficientNet-B3 CNN branch ────────
    # EfficientNet-B3 via timm: model.cnn_branch.backbone.blocks[-1][-1].conv_pwl
    # This is the last pointwise conv before global pooling — richest features.
    try:
        target_layer = model.cnn_branch.backbone.blocks[-1][-1].conv_pwl
        logger.info("Hook target: cnn_branch.backbone.blocks[-1][-1].conv_pwl")
    except (AttributeError, IndexError):
        # Fallback: last layer of the feature extractor
        target_layer = list(model.cnn_branch.backbone.modules())[-1]
        logger.info(f"Hook target (fallback): {type(target_layer).__name__}")

    gradcam = GradCAM(model, target_layer)

    # ── Load test CSV ─────────────────────────────────────────────────────
    test_csv = CFG.SPLITS_DIR / "test_processed.csv"
    test_df  = pd.read_csv(test_csv)
    classes  = CFG.LUNG_CLASSES

    transform = get_val_transforms()
    rng       = random.Random(RANDOM_SEED)

    gradcam_root = CFG.RESULTS_DIR / "gradcam"

    total_saved = 0

    for cls_name in classes:
        cls_idx  = classes.index(cls_name)
        cls_df   = test_df[test_df["label_idx"] == cls_idx].reset_index(drop=True)
        n        = min(SAMPLES_PER_CLASS, len(cls_df))
        idxs     = rng.sample(range(len(cls_df)), n)

        out_dir  = gradcam_root / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n[{cls_name}] Generating {n} Grad-CAM maps ...")

        for i, row_idx in enumerate(idxs):
            row  = cls_df.iloc[row_idx]
            path = row["processed_path"]

            # Load image
            try:
                pil_img = Image.open(path).convert("RGB")
            except Exception as e:
                logger.warning(f"  Skip {path}: {e}")
                continue

            img_tensor = transform(pil_img).unsqueeze(0).to(device)   # (1,3,224,224)
            img_rgb    = tensor_to_rgb(img_tensor.squeeze(0))

            # Grad-CAM
            cam, pred_idx, conf = gradcam.generate(img_tensor)

            pred_cls = classes[pred_idx]
            blended  = overlay_cam(img_rgb, cam)

            save_path = out_dir / f"img_{i+1:02d}_gradcam.png"
            save_gradcam_figure(
                img_rgb, cam, blended,
                true_cls=cls_name, pred_cls=pred_cls,
                conf=conf, save_path=save_path,
            )
            status = "OK" if cls_name == pred_cls else "MISCLASSIFIED"
            logger.info(f"  [{i+1}/{n}] pred={pred_cls} ({conf*100:.1f}%) "
                        f"{status} -> {save_path.name}")
            total_saved += 1

    gradcam.remove()

    logger.info(f"\nGrad-CAM complete. {total_saved} images saved to "
                f"{gradcam_root}")
    logger.info("Next: python src\10_cross_dataset.py  (or: python main.py)")


if __name__ == "__main__":
    main()

