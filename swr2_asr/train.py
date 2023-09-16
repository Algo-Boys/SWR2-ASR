"""Training script for the ASR model."""
import os
from typing import TypedDict

import click
import torch
import torch.nn.functional as F
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from swr2_asr.model_deep_speech import SpeechRecognitionModel
from swr2_asr.utils.data import DataProcessing, MLSDataset, Split
from swr2_asr.utils.decoder import greedy_decoder
from swr2_asr.utils.tokenizer import CharTokenizer

from swr2_asr.utils.loss_scores import cer, wer


class IterMeter:
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
    """Test"""
    print("\nevaluating...")

    # get values from test_args:
    model, device, test_loader, criterion, tokenizer, decoder = test_args.values()

    if decoder == "greedy":
        decoder = greedy_decoder

    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for _data in tqdm(test_loader, desc="Validation Batches"):
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


@click.command()
@click.option(
    "--config_path",
    default="config.yaml",
    help="Path to yaml config file",
    type=click.Path(exists=True),
)
def main(config_path: str):
    """Main function for training the model.

    Gets all configuration arguments from yaml config file.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member

    torch.manual_seed(7)

    with open(config_path, "r", encoding="utf-8") as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    # Create separate dictionaries for each top-level key
    model_config = config_dict.get("model", {})
    training_config = config_dict.get("training", {})
    dataset_config = config_dict.get("dataset", {})
    tokenizer_config = config_dict.get("tokenizer", {})
    checkpoints_config = config_dict.get("checkpoints", {})

    if not os.path.isdir(dataset_config["dataset_root_path"]):
        os.makedirs(dataset_config["dataset_root_path"])

    train_dataset = MLSDataset(
        dataset_config["dataset_root_path"],
        dataset_config["language_name"],
        Split.TRAIN,
        download=dataset_config["download"],
        limited=dataset_config["limited_supervision"],
        size=dataset_config["dataset_percentage"],
    )
    valid_dataset = MLSDataset(
        dataset_config["dataset_root_path"],
        dataset_config["language_name"],
        Split.TEST,
        download=dataset_config["download"],
        limited=dataset_config["limited_supervision"],
        size=dataset_config["dataset_percentage"],
    )

    kwargs = {"num_workers": training_config["num_workers"], "pin_memory": True} if use_cuda else {}

    if tokenizer_config["tokenizer_path"] is None:
        print("Tokenizer not found!")
        if click.confirm("Do you want to train a new tokenizer?", default=True):
            pass
        else:
            return
        tokenizer = CharTokenizer.train(
            dataset_config["dataset_root_path"], dataset_config["language_name"]
        )
    tokenizer = CharTokenizer.from_file(tokenizer_config["tokenizer_path"])

    train_data_processing = DataProcessing("train", tokenizer, {"n_feats": model_config["n_feats"]})
    valid_data_processing = DataProcessing("valid", tokenizer, {"n_feats": model_config["n_feats"]})

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=dataset_config["shuffle"],
        collate_fn=train_data_processing,
        **kwargs,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=training_config["batch_size"],
        shuffle=dataset_config["shuffle"],
        collate_fn=valid_data_processing,
        **kwargs,
    )

    model = SpeechRecognitionModel(
        model_config["n_cnn_layers"],
        model_config["n_rnn_layers"],
        model_config["rnn_dim"],
        tokenizer.get_vocab_size(),
        model_config["n_feats"],
        model_config["stride"],
        model_config["dropout"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), training_config["learning_rate"])
    criterion = nn.CTCLoss(tokenizer.get_blank_token()).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=training_config["learning_rate"],
        steps_per_epoch=int(len(train_loader)),
        epochs=training_config["epochs"],
        anneal_strategy="linear",
    )
    prev_epoch = 0

    if checkpoints_config["model_load_path"] is not None:
        checkpoint = torch.load(checkpoints_config["model_load_path"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        prev_epoch = checkpoint["epoch"]

    iter_meter = IterMeter()

    for epoch in range(prev_epoch + 1, training_config["epochs"] + 1):
        train_args: TrainArgs = {
            "model": model,
            "device": device,
            "train_loader": train_loader,
            "criterion": criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch,
            "iter_meter": iter_meter,
        }

        train_loss = train(train_args)

        test_loss, test_cer, test_wer = 0, 0, 0

        test_args: TestArgs = {
            "model": model,
            "device": device,
            "test_loader": valid_loader,
            "criterion": criterion,
            "tokenizer": tokenizer,
            "decoder": "greedy",
        }

        if training_config["eval_every_n"] != 0 and epoch % training_config["eval_every_n"] == 0:
            test_loss, test_cer, test_wer = test(test_args)

        if checkpoints_config["model_save_path"] is None:
            continue

        if not os.path.isdir(os.path.dirname(checkpoints_config["model_save_path"])):
            os.makedirs(os.path.dirname(checkpoints_config["model_save_path"]))

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
            checkpoints_config["model_save_path"] + str(epoch),
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
