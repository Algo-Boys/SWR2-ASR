"""Class containing utils for the ASR system."""
import os
from enum import Enum
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _extract_tar as extract_archive

from swr2_asr.tokenizer import TokenizerType

train_audio_transforms = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100),
)


# create enum specifiying dataset splits
class MLSSplit(str, Enum):
    """Enum specifying dataset as they are defined in the
    Multilingual LibriSpeech dataset"""

    TRAIN = "train"
    TEST = "test"
    DEV = "dev"


class Split(str, Enum):
    """Extending the MLSSplit class to allow for a custom validatio split"""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    DEV = "dev"


def split_to_mls_split(split_name: Split) -> MLSSplit:
    """Converts the custom split to a MLSSplit"""
    if split_name == Split.VALID:
        return MLSSplit.TRAIN
    else:
        return split_name  # type: ignore


class Sample(TypedDict):
    """Type for a sample in the dataset"""

    waveform: torch.Tensor
    spectrogram: torch.Tensor
    input_length: int
    utterance: torch.Tensor
    utterance_length: int
    sample_rate: int
    speaker_id: str
    book_id: str
    chapter_id: str


class MLSDataset(Dataset):
    """Custom Dataset for reading Multilingual LibriSpeech

    Attributes:
        dataset_path (str):
            path to the dataset
        language (str):
            language of the dataset
        split (Split):
            split of the dataset
        mls_split (MLSSplit):
            split of the dataset as defined in the Multilingual LibriSpeech dataset
        dataset_lookup (list):
            list of dicts containing the speakerid, bookid, chapterid and utterance

    directory structure:
        <dataset_path>
        ├── <language>
        │  ├── train
        │  │  ├── transcripts.txt
        │  │  └── audio
        │  │     └── <speakerid>
        │  │        └── <bookid>
        │  │           └── <speakerid>_<bookid>_<chapterid>.opus / .flac

        each line in transcripts.txt has the following format:
        <speakerid>_<bookid>_<chapterid> <utterance>
    """

    def __init__(
        self,
        dataset_path: str,
        language: str,
        split: Split,
        limited: bool,
        download: bool,
        spectrogram_hparams: dict | None,
    ):
        """Initializes the dataset"""
        self.dataset_path = dataset_path
        self.language = language
        self.file_ext = ".opus" if "opus" in language else ".flac"
        self.mls_split: MLSSplit = split_to_mls_split(split)  # split path on disk
        self.split: Split = split  # split used internally

        if spectrogram_hparams is None:
            self.spectrogram_hparams = {
                "sample_rate": 16000,
                "n_fft": 400,
                "win_length": 400,
                "hop_length": 160,
                "n_mels": 128,
                "f_min": 0,
                "f_max": 8000,
                "power": 2.0,
            }
        else:
            self.spectrogram_hparams = spectrogram_hparams

        self.dataset_lookup = []
        self.tokenizer: type[TokenizerType]

        self._handle_download_dataset(download)
        self._validate_local_directory()
        if limited and (split == Split.TRAIN or split == Split.VALID):
            self.initialize_limited()
        else:
            self.initialize()

    def initialize_limited(self) -> None:
        """Initializes the limited supervision dataset"""
        # get file handles
        # get file paths
        # get transcripts
        # create train or validation split

        handles = set()

        train_root_path = os.path.join(self.dataset_path, self.language, "train")

        # get file handles for 9h
        with open(
            os.path.join(train_root_path, "limited_supervision", "9hr", "handles.txt"),
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                handles.add(line.strip())

        # get file handles for 1h splits
        for handle_path in os.listdir(os.path.join(train_root_path, "limited_supervision", "1hr")):
            if handle_path not in range(0, 6):
                continue
            with open(
                os.path.join(
                    train_root_path, "limited_supervision", "1hr", handle_path, "handles.txt"
                ),
                "r",
                encoding="utf-8",
            ) as file:
                for line in file:
                    handles.add(line.strip())

        # get file paths for handles
        file_paths = []
        for handle in handles:
            file_paths.append(
                os.path.join(
                    train_root_path,
                    "audio",
                    handle.split("_")[0],
                    handle.split("_")[1],
                    handle + self.file_ext,
                )
            )

        # get transcripts for handles
        transcripts = []
        with open(os.path.join(train_root_path, "transcripts.txt"), "r", encoding="utf-8") as file:
            for line in file:
                if line.split("\t")[0] in handles:
                    transcripts.append(line.strip())

        # create train or valid split randomly with seed 42
        if self.split == Split.TRAIN:
            np.random.seed(42)
            indices = np.random.choice(len(file_paths), int(len(file_paths) * 0.8))
            file_paths = [file_paths[i] for i in indices]
            transcripts = [transcripts[i] for i in indices]
        elif self.split == Split.VALID:
            np.random.seed(42)
            indices = np.random.choice(len(file_paths), int(len(file_paths) * 0.2))
            file_paths = [file_paths[i] for i in indices]
            transcripts = [transcripts[i] for i in indices]

        # create dataset lookup
        self.dataset_lookup = [
            {
                "speakerid": path.split("/")[-3],
                "bookid": path.split("/")[-2],
                "chapterid": path.split("/")[-1].split("_")[2].split(".")[0],
                "utterance": utterance.split("\t")[1],
            }
            for path, utterance in zip(file_paths, transcripts, strict=False)
        ]

    def initialize(self) -> None:
        """Initializes the entire dataset

        Reads the transcripts.txt file and creates a lookup table
        """
        transcripts_path = os.path.join(
            self.dataset_path, self.language, self.mls_split, "transcripts.txt"
        )

        with open(transcripts_path, "r", encoding="utf-8") as script_file:
            # read all lines in transcripts.txt
            transcripts = script_file.readlines()
            # split each line into (<speakerid>_<bookid>_<chapterid>, <utterance>)
            transcripts = [line.strip().split("\t", 1) for line in transcripts]  # type: ignore
            utterances = [utterance.strip() for _, utterance in transcripts]  # type: ignore
            identifier = [identifier.strip() for identifier, _ in transcripts]  # type: ignore
            identifier = [path.split("_") for path in identifier]

            if self.split == Split.VALID:
                np.random.seed(42)
                indices = np.random.choice(len(utterances), int(len(utterances) * 0.2))
                utterances = [utterances[i] for i in indices]
                identifier = [identifier[i] for i in indices]
            elif self.split == Split.TRAIN:
                np.random.seed(42)
                indices = np.random.choice(len(utterances), int(len(utterances) * 0.8))
                utterances = [utterances[i] for i in indices]
                identifier = [identifier[i] for i in indices]

            self.dataset_lookup = [
                {
                    "speakerid": path[0],
                    "bookid": path[1],
                    "chapterid": path[2],
                    "utterance": utterance,
                }
                for path, utterance in zip(identifier, utterances, strict=False)
            ]

    def set_tokenizer(self, tokenizer: type[TokenizerType]):
        """Sets the tokenizer"""
        self.tokenizer = tokenizer

    def _handle_download_dataset(self, download: bool) -> None:
        """Download the dataset"""
        if not download:
            print("Download flag not set, skipping download")
            return
        # zip exists:
        if os.path.isfile(os.path.join(self.dataset_path, self.language) + ".tar.gz") and download:
            print(f"Found dataset at {self.dataset_path}. Skipping download")
        # zip does not exist:
        else:
            os.makedirs(self.dataset_path, exist_ok=True)
            url = f"https://dl.fbaipublicfiles.com/mls/{self.language}.tar.gz"

            torch.hub.download_url_to_file(
                url, os.path.join(self.dataset_path, self.language) + ".tar.gz"
            )

        # unzip the dataset
        if not os.path.isdir(os.path.join(self.dataset_path, self.language)):
            print(
                f"Unzipping the dataset at \
                    {os.path.join(self.dataset_path, self.language) + '.tar.gz'}"
            )
            extract_archive(
                os.path.join(self.dataset_path, self.language) + ".tar.gz", overwrite=True
            )
        else:
            print("Dataset is already unzipped, validating it now")
            return

    def _validate_local_directory(self):
        # check if dataset_path exists
        if not os.path.exists(self.dataset_path):
            raise ValueError("Dataset path does not exist")
        if not os.path.exists(os.path.join(self.dataset_path, self.language)):
            raise ValueError("Language not downloaded!")
        if not os.path.exists(os.path.join(self.dataset_path, self.language, self.mls_split)):
            raise ValueError("Split not found in dataset")

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset_lookup)

    def __getitem__(self, idx: int) -> Sample:
        """One sample"""
        if self.tokenizer is None:
            raise ValueError("No tokenizer set")
        # get the utterance
        utterance = self.dataset_lookup[idx]["utterance"]

        # get the audio file
        audio_path = os.path.join(
            self.dataset_path,
            self.language,
            self.mls_split,
            "audio",
            self.dataset_lookup[idx]["speakerid"],
            self.dataset_lookup[idx]["bookid"],
            "_".join(
                [
                    self.dataset_lookup[idx]["speakerid"],
                    self.dataset_lookup[idx]["bookid"],
                    self.dataset_lookup[idx]["chapterid"],
                ]
            )
            + self.file_ext,
        )

        waveform, sample_rate = torchaudio.load(audio_path)  # pylint: disable=no-member

        # resample if necessary
        if sample_rate != self.spectrogram_hparams["sample_rate"]:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.spectrogram_hparams["sample_rate"]
            )
            waveform = resampler(waveform)

        spec = (
            torchaudio.transforms.MelSpectrogram(**self.spectrogram_hparams)(waveform)
            .squeeze(0)
            .transpose(0, 1)
        )

        input_length = spec.shape[0] // 2

        utterance_length = len(utterance)

        utterance = self.tokenizer.encode(utterance)

        utterance = torch.LongTensor(utterance.ids)  # pylint: disable=no-member

        return Sample(
            waveform=waveform,
            spectrogram=spec,
            input_length=input_length,
            utterance=utterance,
            utterance_length=utterance_length,
            sample_rate=self.spectrogram_hparams["sample_rate"],
            speaker_id=self.dataset_lookup[idx]["speakerid"],
            book_id=self.dataset_lookup[idx]["bookid"],
            chapter_id=self.dataset_lookup[idx]["chapterid"],
        )


def collate_fn(samples: list[Sample]) -> dict:
    """Collate function for the dataloader

    pads all tensors within a batch to the same dimensions
    """
    waveforms = []
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for sample in samples:
        waveforms.append(sample["waveform"].transpose(0, 1))
        spectrograms.append(sample["spectrogram"])
        labels.append(sample["utterance"])
        input_lengths.append(sample["spectrogram"].shape[0] // 2)
        label_lengths.append(len(sample["utterance"]))

    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    spectrograms = (
        torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return {
        "waveform": waveforms,
        "spectrogram": spectrograms,
        "input_length": input_lengths,
        "utterance": labels,
        "utterance_length": label_lengths,
    }


if __name__ == "__main__":
    DATASET_PATH = "/Volumes/pherkel/SWR2-ASR"
    LANGUAGE = "mls_german_opus"
    split = Split.TRAIN
    DOWNLOAD = False

    dataset = MLSDataset(DATASET_PATH, LANGUAGE, split, False, DOWNLOAD, None)

    tok = Tokenizer.from_file("data/tokenizers/bpe_tokenizer_german_3000.json")
    dataset.set_tokenizer(tok)


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
