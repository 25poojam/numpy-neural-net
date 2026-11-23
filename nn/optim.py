"""Optimizers: rules for turning gradients into parameter updates."""
import numpy as np


class SGD:
    """Stochastic gradient descent with optional momentum.

    Momentum keeps a running "velocity" so updates keep moving in directions
    that are consistently downhill, and smooth out noisy ones.
    """

    def __init__(self, model, lr=0.01, momentum=0.0):
        self.model, self.lr, self.momentum = model, lr, momentum
        self.velocity = {}

    def step(self):
        for key, p, g in self.model.parameters():
            v = self.velocity.get(key, np.zeros_like(p))
            v = self.momentum * v - self.lr * g
            self.velocity[key] = v
            p += v  # in-place, so the layer's array is updated


class Adam:
    """Adam: per-parameter adaptive learning rates.

    Tracks a moving average of gradients (m) and of squared gradients (v).
    Dividing by sqrt(v) means parameters with large, noisy gradients take
    smaller steps. The bias correction fixes m and v starting at zero.
    """

    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model, self.lr = model, lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self):
        self.t += 1
        for key, p, g in self.model.parameters():
            m = self.b1 * self.m.get(key, np.zeros_like(p)) + (1 - self.b1) * g
            v = self.b2 * self.v.get(key, np.zeros_like(p)) + (1 - self.b2) * g * g
            self.m[key], self.v[key] = m, v
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
