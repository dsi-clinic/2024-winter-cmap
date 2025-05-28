#!/usr/bin/env python
"""
prompted_kc.py
Prompted subclass of KaneCounty that returns a central prompt point.
"""
import sys, os
from pathlib import Path

# Expose repo root so `import data.kc` works
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import numpy as np
from torchgeo.datasets import BoundingBox
from data.kc import KaneCounty

class PromptedKaneCounty(KaneCounty):
    """
    Wraps the GeoDataset KaneCounty to also return a central prompt point.
    """

    def __init__(self, path: str, configs):
        """
        Args:
            path: path to the .gpkg
            configs: (layer_name, label_map, chip_size, dest_crs, res)
        """
        super().__init__(path, configs)
        self.chip_size = configs[2]

    def __getitem__(self, query: BoundingBox):
        """
        Returns the same dict as KaneCounty, plus:
            'point': np.array([x_center, y_center], dtype=int)
        """
        sample = super().__getitem__(query)
        # pixel center in the returned chip
        cx = self.chip_size // 2
        cy = self.chip_size // 2
        sample["point"] = np.array([cx, cy], dtype=np.int32)
        return sample

    def get_labels(self):
        """Return label mapping {name: id}."""
        return self.labels

    def get_inverse_labels(self):
        """Return inverse mapping {id: name}."""
        return self.labels_inverse

    def get_colors(self):
        """Return color mapping {id: (R,G,B,A)}."""
        return self.colors
