"""Utilities for visualizing the training process and results."""

import matplotlib.pyplot as plt
import torch


def plot(epochs, path):
    """Plots the losses over the epochs"""
    losses = list()
    test_losses = list()
    cers = list()
    wers = list()
    for epoch in range(1, epochs + 1):
        current_state = torch.load(path + str(epoch))
        losses.append(current_state["loss"])
        test_losses.append(current_state["test_loss"])
        cers.append(current_state["avg_cer"])
        wers.append(current_state["avg_wer"])

    plt.plot(losses)
    plt.plot(test_losses)
    plt.savefig("losses.svg")
