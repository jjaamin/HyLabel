from __future__ import annotations
import os
from typing import List, Tuple

import numpy as np

WEIGHTS_FILENAME = "edge_sam_3x.pth"
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, WEIGHTS_FILENAME)

DOWNLOAD_URL = (
    "https://huggingface.co/chongzhou/EdgeSAM/resolve/main/weights/"
    + WEIGHTS_FILENAME
)


def is_installed() -> bool:
    try:
        import edge_sam  # noqa: F401
        return True
    except ImportError:
        return False


class EdgeSAMPredictor:
    """Thin wrapper around EdgeSAM's SamPredictor."""

    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        from edge_sam import sam_model_registry, SamPredictor as _SP

        model_type = "edge_sam_3x"
        if model_type not in sam_model_registry:
            model_type = next(iter(sam_model_registry))

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        self._pred = _SP(sam)

    def set_image(self, image_rgb: np.ndarray) -> None:
        self._pred.set_image(image_rgb)

    def predict(
        self,
        points: List[Tuple[int, int]],
        labels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (masks [3,H,W] bool, scores [3])."""
        coords = np.array(points, dtype=np.float32)
        lbls = np.array(labels, dtype=np.int32)
        masks, scores, _ = self._pred.predict(
            point_coords=coords,
            point_labels=lbls,
            multimask_output=True,
        )
        return masks, scores
