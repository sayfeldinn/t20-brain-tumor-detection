"""Create a tiny dummy TorchScript model for contract-testing the API.

Outputs 4 logits over CLASS_NAMES. Delete/overwrite before final packaging:
the graded model must be trained by the team.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # T20/ root on path

import torch
import torch.nn as nn

from app import config


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, len(config.CLASS_NAMES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: 1x3x224x224 in [0,1] -> 1x4 logits (normalization "baked in" = none here)
        feats = self.pool(x).flatten(1)
        return self.fc(feats)


if __name__ == "__main__":
    model = torch.jit.trace(DummyNet(), torch.zeros(1, 3, 224, 224))
    model.save(config.MODEL_PATH)
    print(f"saved dummy {config.MODEL_PATH}")
