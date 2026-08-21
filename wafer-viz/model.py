"""
CNN classifier for wafer defect maps, adapted from the architecture in
MathWorks' "Classify Anomalies on Wafer Defect Maps Using Deep Learning"

example: 
- 4 conv blocks (8/16/32/64 filters) with batchnorm + ReLU + max
    pooling, dropout, then a linear classification head over umbrella defect
    classes. 
- In-house subtypes are not modeled as separate output classes
    (too few verified examples per subtype to be reliable) --> they're found via
    nearest-neighbor search over this network's 2nd-to-last-layer embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

INPUT_SIZE = 48


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess(wafer_map: np.ndarray) -> torch.Tensor:
    """Resize a raw {0,1,2}-valued wafer map to INPUT_SIZE x INPUT_SIZE, normalized to [0,1]."""
    img = Image.fromarray(wafer_map.astype(np.float32))
    img = img.resize((INPUT_SIZE, INPUT_SIZE), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 2.0  # {0,1,2} -> [0,1]
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class WaferCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = ConvBlock(1, 8)
        self.block2 = ConvBlock(8, 16)
        self.block3 = ConvBlock(16, 32)
        self.block4 = ConvBlock(32, 64)
        self.dropout = nn.Dropout(0.3)
        # 48 -> 24 -> 12 -> 6 -> 3, so the last block outputs 64 x 3 x 3
        self.classifier = nn.Linear(64 * 3 * 3, num_classes)
        self.num_classes = num_classes
        self._last_conv_output: torch.Tensor | None = None

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(self.block1(x), 2)
        x = F.max_pool2d(self.block2(x), 2)
        x = F.max_pool2d(self.block3(x), 2)
        x = self.block4(x)
        # Grad-CAM target: the last ReLU activation *before* its pooling, matching
        # MATLAB gradCAM's default reduction-layer choice (finer localization than
        # hooking after the final pool would give: 6x6 here vs 3x3).
        if x.requires_grad:
            x.retain_grad()
        self._last_conv_output = x
        return F.max_pool2d(x, 2)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        feat = self.features(x)
        flat = self.dropout(torch.flatten(feat, 1))
        logits = self.classifier(flat)
        if return_embedding:
            return logits, flat
        return logits


class Classifier:
    """Loads a checkpoint and exposes single-wafer predict/embed/grad_cam."""

    def __init__(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.classes: list[str] = ckpt["classes"]
        self.version: str = ckpt["version"]
        self.model = WaferCNN(num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.dev = device()
        self.model.to(self.dev)

    def _input_tensor(self, wafer_map: np.ndarray) -> torch.Tensor:
        return preprocess(wafer_map).unsqueeze(0).to(self.dev)  # (1, 1, H, W)

    @torch.no_grad()
    def predict(self, wafer_map: np.ndarray) -> dict:
        x = self._input_tensor(wafer_map)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return {
            "class_name": self.classes[idx],
            "class_index": idx,
            "confidence": float(probs[idx]),
            "probs": {c: float(p) for c, p in zip(self.classes, probs)},
        }

    @torch.no_grad()
    def embed(self, wafer_map: np.ndarray) -> np.ndarray:
        x = self._input_tensor(wafer_map)
        _, emb = self.model(x, return_embedding=True)
        return emb[0].cpu().numpy()

    def grad_cam(self, wafer_map: np.ndarray, target_class: int | None = None) -> np.ndarray:
        """Returns an INPUT_SIZE x INPUT_SIZE heatmap in [0,1] for the given (or predicted) class."""
        x = self._input_tensor(wafer_map)
        x.requires_grad_(False)
        self.model.zero_grad()
        logits = self.model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = logits[0, target_class]
        score.backward()

        activations = self.model._last_conv_output  # (1, C, h, w)
        grads = activations.grad  # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam_max = cam.max()
        return cam / cam_max if cam_max > 0 else cam


def save_checkpoint(path: str, model: WaferCNN, classes: list[str], version: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "version": version}, path)


def load_production_classifier(models_dir: str = "models") -> Classifier | None:
    prod_file = Path(models_dir) / "production.json"
    if not prod_file.exists():
        return None
    info = json.loads(prod_file.read_text())
    return Classifier(info["checkpoint"])
