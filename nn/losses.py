"""Loss functions."""
import numpy as np


def softmax(logits):
    # Subtract the row max before exp() for numerical stability:
    # softmax is unchanged by shifting, but exp() of big numbers overflows.
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """Softmax + cross-entropy combined.

    Combining them gives a famously simple gradient:
        dL/dlogits = (softmax(logits) - one_hot(y)) / batch_size
    """

    def forward(self, logits, y):
        self._probs = softmax(logits)
        self._y = y
        n = logits.shape[0]
        # Probability the model assigned to the correct class, per example
        correct = self._probs[np.arange(n), y]
        return float(-np.log(correct + 1e-12).mean())

    def backward(self):
        n = self._probs.shape[0]
        grad = self._probs.copy()
        grad[np.arange(n), self._y] -= 1.0
        return grad / n
