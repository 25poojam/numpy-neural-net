"""Gradient checking: the single most important test for a from-scratch net.

We compare the analytic gradients from backward() against numerical
gradients from finite differences:  (f(x + h) - f(x - h)) / 2h
If backprop is correct, they agree to many decimal places.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn import Adam, Dense, Dropout, ReLU, Sequential, SoftmaxCrossEntropy, softmax  # noqa: E402


def relative_error(a, b):
    return np.abs(a - b).max() / max(1e-8, np.abs(a).max() + np.abs(b).max())


def numerical_grad(f, param, h=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = param[i]
        param[i] = old + h; plus = f()
        param[i] = old - h; minus = f()
        param[i] = old
        grad[i] = (plus - minus) / (2 * h)
    return grad


def make_problem(seed=0):
    rng = np.random.default_rng(seed)
    model = Sequential(Dense(5, 7, rng), ReLU(), Dense(7, 3, rng))
    X = rng.normal(size=(4, 5))
    y = rng.integers(0, 3, size=4)
    return model, X, y


def test_parameter_gradients_match_numerical():
    model, X, y = make_problem()
    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(model.forward(X), y)
    model.backward(loss_fn.backward())
    analytic = {k: g.copy() for k, _, g in model.parameters()}

    def f():
        return loss_fn.forward(model.forward(X), y)

    for key, p, _ in model.parameters():
        num = numerical_grad(f, p)
        assert relative_error(analytic[key], num) < 1e-6, key


def test_input_gradient_matches_numerical():
    model, X, y = make_problem(1)
    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(model.forward(X), y)
    dX = model.backward(loss_fn.backward())
    num = numerical_grad(lambda: loss_fn.forward(model.forward(X), y), X)
    assert relative_error(dX, num) < 1e-6


def test_softmax_rows_sum_to_one_and_is_stable():
    p = softmax(np.array([[1000.0, 1001.0, 1002.0], [-5.0, 0.0, 5.0]]))
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_dropout_is_identity_at_inference():
    d = Dropout(0.5, np.random.default_rng(0))
    x = np.ones((3, 4))
    assert np.array_equal(d.forward(x, training=False), x)


def test_model_can_overfit_tiny_dataset():
    """A sanity check every ML engineer runs: can it memorize 10 examples?"""
    rng = np.random.default_rng(0)
    model = Sequential(Dense(4, 32, rng), ReLU(), Dense(32, 3, rng))
    X, y = rng.normal(size=(10, 4)), rng.integers(0, 3, size=10)
    loss_fn, opt = SoftmaxCrossEntropy(), Adam(model, lr=0.01)
    for _ in range(300):
        loss_fn.forward(model.forward(X), y)
        model.backward(loss_fn.backward())
        opt.step()
    assert (model.predict(X) == y).all()
