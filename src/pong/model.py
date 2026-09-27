"""
The policy network. Two weight matrices, and nothing else.

Defined once, here. A checkpoint is only meaningful next to the exact shape
it was trained against, so a second copy of this class is a correctness
problem rather than duplication: when there were two, they had already
drifted -- one was missing the weight initialisation.
"""

import math

import torch
import torch.nn as nn


# Hidden units, and the flattened 80x80 difference image.
H = 200
D = 80 * 80


# ------------------------------------------------------------
# Policy network
#
# Reference NumPy version:
#
# h = np.dot(W1, x)
# h[h < 0] = 0
# logp = np.dot(W2, h)
# p = sigmoid(logp)
# ------------------------------------------------------------

class Policy(nn.Module):

    def __init__(self, inputs=D, hidden=H):
        super().__init__()

        # No biases: two bare weight matrices.
        self.fc1 = nn.Linear(
            inputs,
            hidden,
            bias=False,
        )

        self.fc2 = nn.Linear(
            hidden,
            1,
            bias=False,
        )

        # Match the reference initialization:
        #
        # W1 = randn(H, D) / sqrt(D)
        # W2 = randn(H)    / sqrt(H)

        nn.init.normal_(
            self.fc1.weight,
            mean=0.0,
            std=1.0 / math.sqrt(inputs),
        )

        nn.init.normal_(
            self.fc2.weight,
            mean=0.0,
            std=1.0 / math.sqrt(hidden),
        )

    def forward(self, x):

        h = self.fc1(x)

        h = torch.relu(h)

        logit = self.fc2(h)

        probability = torch.sigmoid(logit)

        return probability.squeeze()
