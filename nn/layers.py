"""Neural network layers implemented with nothing but NumPy.

Every layer has two methods:
  forward(x)   -> computes the output and remembers what it needs for later
  backward(g)  -> takes dLoss/dOutput and returns dLoss/dInput,
                  storing gradients for its own parameters along the way
"""
import numpy as np


class Layer:
    """Base class. Layers without parameters return empty dicts."""

    def forward(self, x, training=True):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError

    def params(self):
        return {}

    def grads(self):
        return {}


class Dense(Layer):
    """Fully connected layer: y = x @ W + b"""

    def __init__(self, in_features, out_features, rng=None):
        rng = rng or np.random.default_rng()
        # He initialization: keeps activations from shrinking/exploding with ReLU
        self.W = rng.normal(0.0, np.sqrt(2.0 / in_features), (in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x, training=True):
        self._x = x  # cache input; we need it to compute dW
        return x @ self.W + self.b

    def backward(self, grad_out):
        # Chain rule, written out for matrices:
        #   dL/dW = x^T @ dL/dy      (shape: in x out)
        #   dL/db = sum over batch of dL/dy
        #   dL/dx = dL/dy @ W^T      (passed to the previous layer)
        self.dW = self._x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}


class ReLU(Layer):
    """max(0, x). Gradient is 1 where x > 0, otherwise 0."""

    def forward(self, x, training=True):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out):
        return grad_out * self._mask


class Dropout(Layer):
    """Randomly zeroes activations during training to reduce overfitting.

    Uses "inverted dropout": surviving values are scaled up by 1/(1-p) during
    training so nothing needs to change at inference time.
    """

    def __init__(self, p=0.5, rng=None):
        assert 0.0 <= p < 1.0
        self.p = p
        self.rng = rng or np.random.default_rng()
        self._mask = None

    def forward(self, x, training=True):
        if not training or self.p == 0.0:
            self._mask = None
            return x
        self._mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self._mask

    def backward(self, grad_out):
        return grad_out if self._mask is None else grad_out * self._mask
