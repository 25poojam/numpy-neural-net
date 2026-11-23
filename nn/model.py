"""A Sequential model that chains layers together."""
import numpy as np


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad):
        # Backpropagation = walk the layers in reverse, applying the chain rule
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        """Yield (unique_key, param_array, grad_array) for every parameter."""
        for i, layer in enumerate(self.layers):
            grads = layer.grads()
            for name, p in layer.params().items():
                yield f"{i}.{name}", p, grads[name]

    def predict(self, x):
        return self.forward(x, training=False).argmax(axis=1)

    def save(self, path):
        np.savez(path, **{k: p for k, p, _ in self.parameters()})

    def load(self, path):
        data = np.load(path)
        for k, p, _ in self.parameters():
            p[...] = data[k]
