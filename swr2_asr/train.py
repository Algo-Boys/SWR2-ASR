"""Training script for the ASR model."""
import os
from typing import TypedDict

import click
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from swr2_asr.model_deep_speech import HParams, SpeechRecognitionModel
from swr2_asr.utils.data import DataProcessing, MLSDataset, Split
from swr2_asr.utils.decoder import greedy_decoder
from swr2_asr.utils.tokenizer import CharTokenizer

from .utils.loss_scores import cer, wer


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        """step"""
        self.val += 1

    def get(self):
        """get steps"""
        return self.val


class TrainArgs(TypedDict):
    """Type for the arguments of the training function."""

    model: SpeechRecognitionModel
    device: torch.device  # pylint: disable=no-member
    train_loader: DataLoader
    criterion: nn.CTCLoss
    optimizer: optim.AdamW
    scheduler: optim.lr_scheduler.OneCycleLR
    epoch: int
    iter_meter: IterMeter


def train(train_args) -> float:
    """Train
    Args:
        model: model
        device: device type
        train_loader: train dataloader
        criterion: loss function
        optimizer: optimizer
        scheduler: learning rate scheduler
        epoch: epoch number
        iter_meter: iteration meter

    Returns:
        avg_train_loss: avg_train_loss for the epoch

    Information:
        spectrograms: (batch, time, feature)
        labels: (batch, label_length)

        model output: (batch,time, n_class)

    """
    # get values from train_args:
    (
        model,
        device,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        epoch,
        iter_meter,
    ) = train_args.values()

    model.train()
    print(f"training batch {epoch}")
    train_losses = []
    for _data in tqdm(train_loader, desc="Training batches"):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        train_losses.append(loss)
        loss.backward()

        optimizer.step()
        scheduler.step()
        iter_meter.step()
    avg_train_loss = sum(train_losses) / len(train_losses)
    print(f"Train set: Average loss: {avg_train_loss:.2f}")
    return avg_train_loss


class TestArgs(TypedDict):
    """Type for the arguments of the test function."""

    model: SpeechRecognitionModel
    device: torch.device  # pylint: disable=no-member
    test_loader: DataLoader
    criterion: nn.CTCLoss
    tokenizer: CharTokenizer
    decoder: str


def test(test_args: TestArgs) -> tuple[float, float, float]:
    print("\nevaluating...")

    # get values from test_args:
    model, device, test_loader, criterion, tokenizer, decoder = test_args.values()

    if decoder == "greedy":
        decoder = greedy_decoder

    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in enumerate(tqdm(test_loader, desc="Validation Batches")):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = greedy_decoder(
                output.transpose(0, 1), labels, label_lengths, tokenizer
            )
            if i == 1:
                print(f"decoding first sample: {decoded_preds}")
            for j, _ in enumerate(decoded_preds):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    print(
        f"Test set: \
            Average loss: {test_loss:.4f}, \
            Average CER: {avg_cer:4f} \
            Average WER: {avg_wer:.4f}\n"
    )

    return test_loss, avg_cer, avg_wer


def main(
    learning_rate: float,
    batch_size: int,
    epochs: int,
    dataset_path: str,
    language: str,
    limited_supervision: bool,
    model_load_path: str,
    model_save_path: str,
    dataset_percentage: float,
    eval_every: int,
    num_workers: int,
):
    """Main function for training the model.

    Args:
        learning_rate: learning rate for the optimizer
        batch_size: batch size
        epochs: number of epochs to train
        dataset_path: path for the dataset
        language: language of the dataset
        limited_supervision: whether to use only limited supervision
        model_load_path: path to load a model from
        model_save_path: path to save the model to
        dataset_percentage: percentage of the dataset to use
        eval_every: evaluate every n epochs
        num_workers: number of workers for the dataloader
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member

    torch.manual_seed(7)

    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)

    train_dataset = MLSDataset(
        dataset_path,
        language,
        Split.TEST,
        download=True,
        limited=limited_supervision,
        size=dataset_percentage,
    )
    valid_dataset = MLSDataset(
        dataset_path,
        language,
        Split.TRAIN,
        download=False,
        limited=Falimited_supervisionlse,
        size=dataset_percentage,
    )

    # TODO: initialize and possibly train tokenizer if none found

    kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

    hparams = HParams(
        n_cnn_layers=3,
        n_rnn_layers=5,
        rnn_dim=512,
        n_class=tokenizer.get_vocab_size(),
        n_feats=128,
        stride=2,
        dropout=0.1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
    )

    train_data_processing = DataProcessing("train", tokenizer)
    valid_data_processing = DataProcessing("valid", tokenizer)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        collate_fn=train_data_processing,
        **kwargs,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        collate_fn=valid_data_processing,
        **kwargs,
    )

    model = SpeechRecognitionModel(
        hparams["n_cnn_layers"],
        hparams["n_rnn_layers"],
        hparams["rnn_dim"],
        hparams["n_class"],
        hparams["n_feats"],
        hparams["stride"],
        hparams["dropout"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), hparams["learning_rate"])
    criterion = nn.CTCLoss(tokenizer.get_blank_token()).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams["learning_rate"],
        steps_per_epoch=int(len(train_loader)),
        epochs=hparams["epochs"],
        anneal_strategy="linear",
    )
    prev_epoch = 0

    if model_load_path is not None:
        checkpoint = torch.load(model_load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        prev_epoch = checkpoint["epoch"]

    iter_meter = IterMeter()
    if not os.path.isdir(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    for epoch in range(prev_epoch + 1, epochs + 1):
        train_args: TrainArgs = dict(
            model=model,
            device=device,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            iter_meter=iter_meter,
        )

        train_loss = train(train_args)

        test_loss, test_cer, test_wer = 0, 0, 0

        test_args: TestArgs = dict(
            model=model,
            device=device,
            test_loader=valid_loader,
            criterion=criterion,
            tokenizer=tokenizer,
            decoder="greedy",
        )

        if epoch % eval_every == 0:
            test_loss, test_cer, test_wer = test(test_args)

        if model_save_path is None:
            continue

        if not os.path.isdir(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "avg_cer": test_cer,
                "avg_wer": test_wer,
            },
            model_save_path + str(epoch),
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
