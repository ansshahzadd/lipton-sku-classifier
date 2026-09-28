"""
Minimal reconstruction of the MLP head class the guide's checkpoints
(REJECT_GATE_DINOV3_MLP_V3.pt, SKU_32CLASS_DINOV3_MLP.pt) were pickled
with -- torch.load() needs a class named exactly `model_heads.MLPHead` to
unpickle them, and the training code that originally defined it isn't part
of this repo.

Loading an existing checkpoint never calls __init__: torch restores
`self.net` (an nn.Sequential of Linear/ReLU/Dropout layers) directly from
the saved state, so this class only needs to exist and know how to run
forward() on it. It is not meant for training a new head from scratch.
"""

import torch.nn as nn


class MLPHead(nn.Module):
    def forward(self, x):
        return self.net(x)
