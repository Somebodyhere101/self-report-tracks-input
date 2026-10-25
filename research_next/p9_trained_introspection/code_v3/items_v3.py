"""Paper 9 v3: seeded random unit directions (tamper control), as in v1's items.random_unit."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v3lib import seeded  # noqa: E402


def random_unit(d, key):
    g = torch.Generator().manual_seed(int(seeded("rand", key).random() * 2 ** 31))
    v = torch.randn(d, generator=g)
    return v / v.norm()
