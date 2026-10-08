import os
import sys
import numpy as np

# Apply numpy compatibility shims for scipy/sklearn
for attr in ["long", "ulong"]:
    if not hasattr(np, attr):
        setattr(np, attr, int)

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
