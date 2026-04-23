"""dann-domain-adaptation: DANN with Gradient Reversal Layer."""
from .grl import GradientReversalLayer, GradientReversalFunction
from .dann_model import DANNModel

__version__ = "0.1.0"
__all__ = ["GradientReversalLayer", "GradientReversalFunction", "DANNModel"]
