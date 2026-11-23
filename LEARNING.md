# Study Guide: Neural Network from Scratch

Work through this in order. Aim to be able to **explain each step out loud**, because interviewers will ask about it.

---

## 1. The big picture (read first)

Training a neural net is a loop:

1. **Forward pass**: push a batch of inputs through the layers to get predictions (logits).
2. **Loss**: measure how wrong the predictions are (cross-entropy).
3. **Backward pass**: compute how much each weight contributed to the error (gradients).
4. **Update**: nudge every weight a little in the direction that lowers the loss (optimizer).

Find these four steps in the inner loop of `train.py`. They're 4 lines.

## 2. Dense layer (`nn/layers.py`)

`y = x @ W + b`. Shapes: `x` is (batch, in), `W` is (in, out), `y` is (batch, out).

The backward pass needs three gradients:

| Gradient | Formula | Why |
|---|---|---|
| `dW` | `x.T @ grad_out` | Each weight's effect is scaled by the input it multiplies |
| `db` | `grad_out.sum(0)` | Bias is added to every example, so sum over the batch |
| `dx` | `grad_out @ W.T` | Sent back to the previous layer |

**Exercise:** check the shapes of all three by hand. Shape-checking is how you derive these formulas without calculus.

**Why He initialization?** If weights are too big, activations explode layer after layer; too small, they vanish. `sqrt(2/in)` keeps the variance stable through ReLU layers.

## 3. ReLU and Dropout

- ReLU's derivative is 1 for positive inputs and 0 otherwise, so backward just masks the gradient.
- Dropout randomly zeroes neurons during training so the network can't rely on any single one. We divide by `(1-p)` during training ("inverted dropout") so the expected value is unchanged and inference needs no special handling.

**Interview question:** *Why is dropout turned off at test time?* (You want deterministic predictions using the full network.)

## 4. Softmax + cross-entropy (`nn/losses.py`)

- Softmax turns logits into probabilities.
- Cross-entropy is `-log(probability of the correct class)`.
- **Stability trick:** subtract the max logit before `exp()`. Same answer, no overflow. The test `test_softmax_rows_sum_to_one_and_is_stable` feeds it 1000+ values to prove this.
- **The beautiful result:** the gradient of the combined function is `probs - one_hot(y)`. Try deriving it on paper; it's a classic interview question.

## 5. Optimizers (`nn/optim.py`)

- **SGD:** `w -= lr * grad`.
- **Momentum:** keep a velocity; consistent directions speed up, noisy ones cancel out.
- **Adam:** tracks the average gradient (`m`) and average squared gradient (`v`). Dividing by `sqrt(v)` gives each parameter its own effective learning rate. Bias correction matters early because `m` and `v` start at zero.

**Experiment:** run `python train.py --optimizer sgd --lr 0.05` vs the default Adam. Compare how fast validation accuracy climbs.

## 6. Gradient checking (`tests/test_gradients.py`)

The numerical derivative `(f(x+h) - f(x-h)) / 2h` is slow but obviously correct. If it matches your backprop, your backprop is right.

**Exercise:** break something on purpose. Delete `.T` in `Dense.backward`, or forget `/ n` in the loss gradient, and run `pytest`. Watch the test catch it. This is the best way to build intuition.

`test_model_can_overfit_tiny_dataset` is a real-world habit: before training on big data, confirm your model can memorize 10 examples. If it can't, there's a bug.

## 7. Train / validation / test

- **Train:** used to update weights.
- **Validation:** used to pick the best epoch and hyperparameters.
- **Test:** used *once* at the end. If you tune on it, your reported score is optimistic.

Look at `assets/training_curves.png`: train accuracy reaches ~100% while validation plateaus around 98%. That gap is mild overfitting. **Experiment:** set `--dropout 0` and see if the gap grows.

---

## Interview questions you should now be able to answer

1. Walk me through backpropagation for a single dense layer.
2. Why do we subtract the max in softmax?
3. What's the difference between SGD with momentum and Adam?
4. What is dropout and why does it help?
5. How would you know your gradients are correct?
6. Why keep a separate validation set?

## Stretch goals (good commits to add yourself)

1. Add a `Sigmoid` or `Tanh` layer plus a gradient-check test for it.
2. Add L2 weight decay to the optimizers.
3. Add a learning-rate schedule (e.g. halve the LR every 15 epochs).
4. Print a confusion matrix: which digits get confused most?

Adding your own commits is important. A portfolio where every project was pushed in one commit looks copied. Real history with your own improvements looks like real work.
