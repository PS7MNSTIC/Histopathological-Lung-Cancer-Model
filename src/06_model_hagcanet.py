"""
06_model_hagcanet.py
====================
HAGCA-Net: Hybrid Adaptive Graph Context Attention Network
for histopathological lung cancer classification.

Architecture:
  1. CNN Branch      — EfficientNet-B3 (local texture features)
  2. Transformer     — Swin-Base (global context features)
  3. Graph Learning  — spatial relationship modelling on CNN feature map
  4. Adaptive Fusion — learned weighting of the three branches
  5. Context Attention — multi-head self-attention over fused token
  6. Classifier      — Dropout → Linear → 3 classes

All novel modules (3–5) implemented in pure PyTorch (no torch-geometric).

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\06_model_hagcanet.py        # smoke-tests on GPU
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import timm

sys.path.insert(0, str(Path(__file__).parent))
from config import CFG, setup_device


# ════════════════════════════════════════════════════════════════════════════
#  1.  CNN Branch  —  EfficientNet-B3
#      Returns: global-pooled features  (B, CNN_OUT_DIM)
#               raw feature map         (B, feat_ch, H, W)  for graph module
# ════════════════════════════════════════════════════════════════════════════

class CNNBranch(nn.Module):
    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        super().__init__()
        # features_only=True → returns list of intermediate feature maps
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            features_only=True,
        )
        # Gradient checkpointing: recomputes activations on backward pass,
        # trading ~40% extra compute for ~60% less activation memory.
        self.backbone.set_grad_checkpointing(enable=True)

        # Channel count of the LAST feature stage (stage 4 / index -1)
        last_ch = self.backbone.feature_info[-1]["num_chs"]   # 384 for EffNet-B3
        self.feat_dim = last_ch

        self.pool = nn.AdaptiveAvgPool2d(1)          # → (B, last_ch, 1, 1)
        self.proj = nn.Sequential(
            nn.Flatten(),                             # → (B, last_ch)
            nn.Linear(last_ch, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        feats    = self.backbone(x)      # list of feature maps at each stage
        feat_map = feats[-1]             # (B, 384, 7, 7) for 224px input
        pooled   = self.pool(feat_map)   # (B, 384, 1, 1)
        return self.proj(pooled), feat_map   # (B, out_dim), (B, 384, 7, 7)


# ════════════════════════════════════════════════════════════════════════════
#  2.  Transformer Branch  —  Swin-Base
#      Returns: global-pooled features  (B, TRANS_OUT_DIM)
# ════════════════════════════════════════════════════════════════════════════

class TransformerBranch(nn.Module):
    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,       # remove classification head
            global_pool="avg",   # global average pool → (B, 1024)
        )
        # Swin-Base is the memory bottleneck (87M params, large attention maps).
        # Gradient checkpointing halves peak activation memory at ~30% compute cost.
        self.backbone.set_grad_checkpointing(enable=True)
        swin_dim = self.backbone.num_features   # 1024 for Swin-Base

        self.proj = nn.Sequential(
            nn.Linear(swin_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        feats = self.backbone(x)     # (B, 1024)
        return self.proj(feats)      # (B, out_dim)


# ════════════════════════════════════════════════════════════════════════════
#  3.  Graph Learning Module  (pure PyTorch — no torch-geometric needed)
#
#      Treats each spatial position of the CNN feature map as a graph node.
#      For a 7×7 map: 49 nodes, each with `in_dim` features.
#
#      Learnable adjacency + 2-layer GCN aggregation.
#      Returns: graph-pooled features  (B, out_dim)
# ════════════════════════════════════════════════════════════════════════════

class GraphLearningModule(nn.Module):
    def __init__(
        self,
        in_dim:     int = 384,
        hidden_dim: int = 256,
        out_dim:    int = 256,
        n_nodes:    int = 49,     # 7×7 spatial locations
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes

        # Learnable node-to-node affinity (symmetric initialised as identity)
        self.adj_raw = nn.Parameter(torch.eye(n_nodes))

        # Two-layer GCN
        self.gc1  = nn.Linear(in_dim,     hidden_dim, bias=False)
        self.bn1  = nn.BatchNorm1d(hidden_dim)
        self.gc2  = nn.Linear(hidden_dim, out_dim,    bias=False)
        self.bn2  = nn.BatchNorm1d(out_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def _normalise_adj(self) -> torch.Tensor:
        """Symmetric softmax + degree-normalisation: Â = D^{-1/2} A D^{-1/2}"""
        A  = torch.softmax(self.adj_raw, dim=-1)          # row-wise, (N, N)
        A  = (A + A.T) / 2                                # symmetrise
        D  = A.sum(dim=-1).clamp(min=1e-6)                # degree vector
        D_inv_sqrt = D.pow(-0.5).diag()                   # D^{-1/2}
        return D_inv_sqrt @ A @ D_inv_sqrt                 # (N, N)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        # feat_map: (B, C, H, W)
        B, C, H, W = feat_map.shape
        N = H * W                      # number of graph nodes
        assert N == self.n_nodes, f"Expected {self.n_nodes} nodes, got {N}"

        # (B, C, H, W) → (B, N, C) — each pixel is a node
        x = feat_map.flatten(2).transpose(1, 2)   # (B, N, C)

        A_hat = self._normalise_adj()              # (N, N)

        # GCN layer 1: H' = GELU( BN( A_hat * X * W1 ) )
        x = self.gc1(x)                            # (B, N, hidden_dim)
        x = torch.matmul(A_hat, x)                # (B, N, hidden_dim)  — graph aggregate
        # BatchNorm1d expects (B, C) or (B, C, L) — reshape
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)

        # GCN layer 2: H'' = GELU( BN( A_hat * H' * W2 ) )
        x = self.gc2(x)                            # (B, N, out_dim)
        x = torch.matmul(A_hat, x)                # graph aggregate
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)

        # Global mean pool over graph nodes → (B, out_dim)
        return x.mean(dim=1)


# ════════════════════════════════════════════════════════════════════════════
#  4.  Adaptive Fusion Module
#
#      Learns per-sample attention weights over the three feature branches,
#      then computes their weighted sum in a shared embedding space.
# ════════════════════════════════════════════════════════════════════════════

class AdaptiveFusionModule(nn.Module):
    def __init__(
        self,
        cnn_dim:   int = 512,
        trans_dim: int = 512,
        graph_dim: int = 256,
        out_dim:   int = 512,
    ):
        super().__init__()
        total = cnn_dim + trans_dim + graph_dim

        # Gate: scalar weight for each of the 3 branches
        self.gate = nn.Sequential(
            nn.Linear(total, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1),       # weights sum to 1
        )

        # Project each branch to a common out_dim
        self.proj_cnn   = nn.Linear(cnn_dim,   out_dim, bias=False)
        self.proj_trans = nn.Linear(trans_dim, out_dim, bias=False)
        self.proj_graph = nn.Linear(graph_dim, out_dim, bias=False)

        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        f_cnn:   torch.Tensor,   # (B, cnn_dim)
        f_trans: torch.Tensor,   # (B, trans_dim)
        f_graph: torch.Tensor,   # (B, graph_dim)
    ) -> torch.Tensor:
        combined = torch.cat([f_cnn, f_trans, f_graph], dim=-1)  # (B, total)
        w = self.gate(combined)                                    # (B, 3)
        w0, w1, w2 = w[:, 0:1], w[:, 1:2], w[:, 2:3]            # each (B, 1)

        fused = (
            w0 * self.proj_cnn(f_cnn) +
            w1 * self.proj_trans(f_trans) +
            w2 * self.proj_graph(f_graph)
        )
        return self.norm(fused)   # (B, out_dim)


# ════════════════════════════════════════════════════════════════════════════
#  5.  Context Attention Module
#
#      Learnable query tokens attend over the fused feature to produce
#      a context-aware representation highlighting clinically relevant
#      aspects of the joint embedding.
# ════════════════════════════════════════════════════════════════════════════

class ContextAttention(nn.Module):
    def __init__(
        self,
        dim:       int = 512,
        n_heads:   int = 8,
        n_queries: int = 4,
        dropout:   float = 0.1,
    ):
        super().__init__()
        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)

        # Cross-attention: queries attend over fused feature (1 key/value token)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)

        # Feed-forward refinement
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)  — fused feature
        B = x.size(0)
        kv = x.unsqueeze(1)                          # (B, 1, D)
        q  = self.queries.expand(B, -1, -1)          # (B, n_queries, D)

        # Cross-attention
        attn_out, _ = self.cross_attn(q, kv, kv)    # (B, n_queries, D)
        attn_out = self.norm1(attn_out + q)          # residual

        # Feed-forward
        attn_out = self.norm2(attn_out + self.ff(attn_out))

        # Aggregate query tokens → single context vector
        return attn_out.mean(dim=1)                  # (B, D)


# ════════════════════════════════════════════════════════════════════════════
#  6.  HAGCA-Net  —  Full Model
# ════════════════════════════════════════════════════════════════════════════

class HAGCANet(nn.Module):
    """
    Hybrid Adaptive Graph Context Attention Network.

    Forward pass:
      x (B, 3, 224, 224)
        → CNN branch       → f_cnn   (B, 512)
        → Graph module     → f_graph (B, 256)   [uses CNN feature map]
        → Swin branch      → f_trans (B, 512)
        → Adaptive Fusion  → f_fused (B, 512)
        → Context Attention→ f_ctx   (B, 512)
        → Classifier       → logits  (B, 3)
    """

    def __init__(
        self,
        num_classes: int  = CFG.NUM_CLASSES,
        pretrained:  bool = CFG.PRETRAINED,
        dropout:     float = CFG.DROPOUT_RATE,
    ):
        super().__init__()

        # ── branches ──────────────────────────────────────────────────────
        self.cnn_branch   = CNNBranch(out_dim=CFG.CNN_OUT_DIM, pretrained=pretrained)
        self.trans_branch = TransformerBranch(out_dim=CFG.TRANS_OUT_DIM, pretrained=pretrained)

        cnn_feat_ch = self.cnn_branch.feat_dim    # 384 for EfficientNet-B3

        # ── novel modules ──────────────────────────────────────────────────
        self.graph_module = GraphLearningModule(
            in_dim=cnn_feat_ch,
            hidden_dim=CFG.GNN_HIDDEN_DIM,
            out_dim=CFG.GNN_HIDDEN_DIM,
            n_nodes=49,                           # 7×7 spatial grid at 224px input
            dropout=dropout,
        )

        self.fusion = AdaptiveFusionModule(
            cnn_dim=CFG.CNN_OUT_DIM,
            trans_dim=CFG.TRANS_OUT_DIM,
            graph_dim=CFG.GNN_HIDDEN_DIM,
            out_dim=CFG.FUSION_DIM,
        )

        self.context_attn = ContextAttention(
            dim=CFG.FUSION_DIM,
            n_heads=8,
            n_queries=4,
            dropout=dropout,
        )

        # ── classifier ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(CFG.FUSION_DIM, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN branch — also returns raw feature map for graph learning
        f_cnn, feat_map = self.cnn_branch(x)     # (B,512), (B,384,7,7)

        # Graph learning on spatial CNN features
        f_graph = self.graph_module(feat_map)     # (B, 256)

        # Swin Transformer branch
        f_trans = self.trans_branch(x)            # (B, 512)

        # Adaptive fusion of all three streams
        f_fused = self.fusion(f_cnn, f_trans, f_graph)   # (B, 512)

        # Context attention — highlights clinically relevant features
        f_ctx = self.context_attn(f_fused)        # (B, 512)

        # Final classification
        return self.classifier(f_ctx)             # (B, 3)

    # ── Backbone freezing helpers ──────────────────────────────────────────

    def freeze_backbones(self):
        """
        Freeze CNN + Transformer backbone weights.
        Only the novel modules (Graph, Fusion, ContextAttn, Classifier) train.
        Use in Phase 1 to save memory and stabilise novel-module training.
        """
        for p in self.cnn_branch.backbone.parameters():
            p.requires_grad = False
        for p in self.trans_branch.backbone.parameters():
            p.requires_grad = False
        # Keep projection heads trainable
        for p in self.cnn_branch.proj.parameters():
            p.requires_grad = True
        for p in self.trans_branch.proj.parameters():
            p.requires_grad = True

    def unfreeze_backbones(self):
        """Unfreeze all parameters for full fine-tuning in Phase 2."""
        for p in self.parameters():
            p.requires_grad = True

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def model_summary(model: nn.Module, device: torch.device):
    """Print per-module parameter counts."""
    params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"  HAGCA-Net Model Summary")
    print(f"{'='*60}")
    modules = {
        "CNN Branch (EfficientNet-B3)": model.cnn_branch,
        "Transformer Branch (Swin-Base)": model.trans_branch,
        "Graph Learning Module":  model.graph_module,
        "Adaptive Fusion Module": model.fusion,
        "Context Attention":      model.context_attn,
        "Classifier":             model.classifier,
    }
    for name, mod in modules.items():
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:<35} {n/1e6:6.2f} M params")
    print(f"{'─'*60}")
    print(f"  {'Total':<35} {params['total']/1e6:6.2f} M params")
    print(f"  {'Trainable':<35} {params['trainable']/1e6:6.2f} M params")
    print(f"{'='*60}\n")


# ════════════════════════════════════════════════════════════════════════════
#  Smoke test
# ════════════════════════════════════════════════════════════════════════════

def smoke_test(device: torch.device):
    import time

    print(f"\n[Smoke Test] Building HAGCA-Net on {device} ...")
    model = HAGCANet(
        num_classes=CFG.NUM_CLASSES,
        pretrained=True,
        dropout=CFG.DROPOUT_RATE,
    ).to(device)

    model_summary(model, device)

    # Forward pass with a random batch
    bs = 4     # keep small for the smoke test — full bs=64 is for training
    x  = torch.randn(bs, 3, 224, 224, device=device)

    model.eval()
    print(f"[Smoke Test] Forward pass — batch_size={bs}, input={tuple(x.shape)}")
    t0 = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=CFG.AMP and device.type == "cuda"):
            logits = model(x)
    elapsed = time.time() - t0

    print(f"[Smoke Test] Output: {tuple(logits.shape)}  |  "
          f"time={elapsed*1000:.1f} ms  |  device={logits.device}")
    assert logits.shape == (bs, CFG.NUM_CLASSES), \
        f"Expected ({bs}, {CFG.NUM_CLASSES}), got {tuple(logits.shape)}"

    # VRAM usage
    if device.type == "cuda":
        vram_mb = torch.cuda.memory_allocated(device) / 1e6
        print(f"[Smoke Test] VRAM allocated: {vram_mb:.1f} MB")

    print("[Smoke Test] PASSED ✓\n")
    return model


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_device()
    device = torch.device(CFG.DEVICE)

    print("\n" + "=" * 60)
    print("  STEP 6: HAGCA-Net Model Design")
    print("=" * 60)
    print("  CNN backbone  :", CFG.CNN_BACKBONE)
    print("  Trans backbone:", CFG.TRANSFORMER_BACKBONE)
    print("  CNN out dim   :", CFG.CNN_OUT_DIM)
    print("  Trans out dim :", CFG.TRANS_OUT_DIM)
    print("  Fusion dim    :", CFG.FUSION_DIM)
    print("  GNN hidden    :", CFG.GNN_HIDDEN_DIM)
    print("  Num classes   :", CFG.NUM_CLASSES)
    print("  Classes       :", CFG.LUNG_CLASSES)
    print("=" * 60)

    model = smoke_test(device)

    print("[OK] 06_model_hagcanet.py complete.")
    print("     Import HAGCANet from this file in 07_train.py")
    print("     Next: python src\07_train.py  (or: python main.py)")

