"""Train the from-scratch network on handwritten digits.

Usage:
    python train.py                 # default settings
    python train.py --epochs 50 --optimizer sgd --lr 0.05
"""
import argparse

import numpy as np
from sklearn.datasets import load_digits  # used ONLY to load data
from sklearn.model_selection import train_test_split

from nn import SGD, Adam, Dense, Dropout, ReLU, Sequential, SoftmaxCrossEntropy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true", help="save training curves to training_curves.png")
    return p.parse_args()


def load_data(seed):
    digits = load_digits()  # 1,797 images of 8x8 pixels, labels 0-9
    X = digits.data / 16.0  # pixel values are 0..16; scale to 0..1
    y = digits.target
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.15, random_state=seed, stratify=y_tmp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(n_in, hidden, n_out, dropout, rng):
    return Sequential(
        Dense(n_in, hidden, rng), ReLU(), Dropout(dropout, rng),
        Dense(hidden, hidden, rng), ReLU(), Dropout(dropout, rng),
        Dense(hidden, n_out, rng),
    )


def accuracy(model, X, y):
    return float((model.predict(X) == y).mean())


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.seed)
    print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    model = build_model(X_train.shape[1], args.hidden, 10, args.dropout, rng)
    loss_fn = SoftmaxCrossEntropy()
    opt = Adam(model, lr=args.lr) if args.optimizer == "adam" else SGD(model, lr=args.lr, momentum=0.9)

    history = {"loss": [], "train_acc": [], "val_acc": []}
    best_val, best_path = -1.0, "best_model.npz"

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(X_train))  # shuffle every epoch
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            logits = model.forward(X_train[idx], training=True)
            losses.append(loss_fn.forward(logits, y_train[idx]))
            model.backward(loss_fn.backward())
            opt.step()

        tr_acc, va_acc = accuracy(model, X_train, y_train), accuracy(model, X_val, y_val)
        history["loss"].append(np.mean(losses))
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        if va_acc > best_val:  # keep the checkpoint that does best on validation
            best_val = va_acc
            model.save(best_path)
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  train {tr_acc:.3f}  val {va_acc:.3f}")

    model.load(best_path)
    test_acc = accuracy(model, X_test, y_test)
    print(f"\nBest val accuracy: {best_val:.3f}")
    print(f"Test accuracy:     {test_acc:.3f}  (touched only once, at the very end)")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.plot(history["loss"]); a1.set_title("Training loss"); a1.set_xlabel("epoch")
        a2.plot(history["train_acc"], label="train"); a2.plot(history["val_acc"], label="validation")
        a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend()
        fig.tight_layout(); fig.savefig("training_curves.png", dpi=120)
        print("Saved training_curves.png")


if __name__ == "__main__":
    main()
