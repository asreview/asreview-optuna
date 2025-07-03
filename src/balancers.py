import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.class_weight import compute_sample_weight as _compute_sample_weight

class DynamicLossBalancer(BaseEstimator):
    """Dynamic Balancer

    Parameters
    ----------
    ratio : float
        Maximum weighting factor applied to minority class.
    window_size : int
        Number of recent samples to compute pseudo-derivative.
    activation : str
        Activation function to shape the derivative response. One of: 'linear', 'sigmoid', 'tanh'.
    initial_weights : dict
        Initial static weights per class, e.g., {0: 1.0, 1: 1.0}.
    """

    name = "dynamic balancer"
    label = "Dynamic Balanced Sample Weight"

    def __init__(self, ratio=1.0, window_size=10, activation='linear', a=1.0):
        self.ratio = ratio
        self.window_size = window_size
        self.activation = activation
        self.a = a

    def _activation_fn(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def compute_sample_weight(self, y):
        if len(set(y)) != 2:
            raise ValueError("Only binary classification is supported.")
        
        slope = (np.sum(y[-min(self.window_size, len(y)):])) / min(self.window_size, len(y))

        base_class_0_weight = sum(y == 1) / (self.ratio * sum(y == 0))
        act_base_class_0_weight = self._activation_fn(base_class_0_weight)

        # Compute dynamic weight: interpolates from 1.0 to ratio
        class_1_weight = 1.0
        class_0_weight = self.ratio * (self.a + (slope * act_base_class_0_weight))

        weights = _compute_sample_weight(
            class_weight={0: class_0_weight, 1: class_1_weight}, y=y
        )
        return weights * (len(y) / np.sum(weights))  # normalize sum of weights to sample count