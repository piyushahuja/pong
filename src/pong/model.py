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
#     6400 -> 200 (ReLU) -> 1 (sigmoid) -> P(move up)
#
# Two matrix multiplications and two nonlinearities. That is the
# entire agent.
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

        # Scale the initial weights by 1/sqrt(inputs) so a layer's output
        # starts at roughly the same scale regardless of how wide it is.

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
