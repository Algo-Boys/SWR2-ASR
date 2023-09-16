"""Utilities for visualizing the training process and results."""

import matplotlib.pyplot as plt
import torch


def plot(path):
    """Plots the losses over the epochs"""
    train_losses = []
    test_losses = []
    cers = []
    wers = []

    epoch = 5
    while True:
        try:
            current_state = torch.load(path + str(epoch), map_location=torch.device("cpu"))
        except FileNotFoundError:
            break
        train_losses.append((epoch, current_state["train_loss"].item()))
        test_losses.append((epoch, current_state["test_loss"]))
        cers.append((epoch, current_state["avg_cer"]))
        wers.append((epoch, current_state["avg_wer"]))
        epoch += 5

    plt.plot(*zip(*train_losses), label="train_loss")
    plt.plot(*zip(*test_losses), label="test_loss")
    plt.plot(*zip(*cers), label="cer")
    plt.plot(*zip(*wers), label="wer")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.title("Model performance for 5n epochs")
    plt.legend()
    plt.savefig("losses.svg")


if __name__ == "__main__":
    plot("data/runs/epoch")
