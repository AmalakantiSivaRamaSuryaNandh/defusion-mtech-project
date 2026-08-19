"""Self-supervised common/unique decomposition for educational image fusion."""

from .baselines import FUSION_METHODS, fuse
from .metrics import fusion_metrics

__all__ = ["FUSION_METHODS", "fuse", "fusion_metrics"]
__version__ = "0.1.0"
