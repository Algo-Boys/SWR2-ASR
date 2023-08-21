"""Class containing utils for the ASR system."""
from dataclasses import dataclass
import os
from AudioLoader.speech import MultilingualLibriSpeech
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
from enum import Enum

from tokenizers import Tokenizer
from swr2_asr.tokenizer import CharTokenizer


# create enum specifiying dataset splits
class MLSSplit(str, Enum):
    """Enum specifying dataset as they are defined in the
    Multilingual LibriSpeech dataset"""

    train = "train"
    test = "test"
    dev = "dev"


class Split(str, Enum):
    """Extending the MLSSplit class to allow for a custom validatio split"""

    train = "train"
    valid = "valid"
    test = "test"
    dev = "dev"


def split_to_mls_split(split: Split) -> MLSSplit:
    """Converts the custom split to a MLSSplit"""
    if split == Split.valid:
        return MLSSplit.train
    else:
        return split  # type: ignore


@dataclass
class Sample:
    """Dataclass for a sample in the dataset"""

    waveform: torch.Tensor
    spectrogram: torch.Tensor
    utterance: str
    sample_rate: int
    speaker_id: str
    book_id: str
    chapter_id: str


def tokenizer_factory(tokenizer_path: str, tokenizer_type: str = "BPE"):
    """Factory for Tokenizer class

    Args:
        tokenizer_type (str, optional): Type of tokenizer to use. Defaults to "BPE".

    Returns:
        nn.Module: Tokenizer class
    """
    if tokenizer_type == "BPE":
        return Tokenizer.from_file(tokenizer_path)
    elif tokenizer_type == "char":
        return CharTokenizer.from_file(tokenizer_path)


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

    def __init__(self, dataset_path: str, language: str, split: Split, download: bool):
        """Initializes the dataset"""
        self.dataset_path = dataset_path
        self.language = language
        self.file_ext = ".opus" if "opus" in language else ".flac"
        self.mls_split: MLSSplit = split_to_mls_split(split)  # split path on disk
        self.split: Split = split  # split used internally
        self.dataset_lookup = []

        self._handle_download_dataset(download)
        self._validate_local_directory()

        transcripts_path = os.path.join(
            dataset_path, language, self.mls_split, "transcripts.txt"
        )

        with open(transcripts_path, "r", encoding="utf-8") as script_file:
            # read all lines in transcripts.txt
            transcripts = script_file.readlines()
            # split each line into (<speakerid>_<bookid>_<chapterid>, <utterance>)
            transcripts = [line.strip().split("\t", 1) for line in transcripts]
            utterances = [utterance.strip() for _, utterance in transcripts]
            identifier = [identifier.strip() for identifier, _ in transcripts]
            identifier = [path.split("_") for path in identifier]

            self.dataset_lookup = [
                {
                    "speakerid": path[0],
                    "bookid": path[1],
                    "chapterid": path[2],
                    "utterance": utterance,
                }
                for path, utterance in zip(identifier, utterances)
            ]

        # save dataset_lookup as list of dicts, where each dict contains
        # the speakerid, bookid and chapterid, as well as the utterance
        # we can then use this to map the utterance to the audio file

    def _handle_download_dataset(self, download: bool):
        """Download the dataset"""
        if (
            not os.path.exists(os.path.join(self.dataset_path, self.language))
            and download
        ):
            os.makedirs(self.dataset_path)
            url = f"https://dl.fbaipublicfiles.com/mls/{self.language}.tar.gz"

            torch.hub.download_url_to_file(url, self.dataset_path)
        elif (
            not os.path.exists(os.path.join(self.dataset_path, self.language))
            and not download
        ):
            raise ValueError("Dataset not found. Set download to True to download it")

    def _validate_local_directory(self):
        # check if dataset_path exists
        if not os.path.exists(self.dataset_path):
            raise ValueError("Dataset path does not exist")
        if not os.path.exists(os.path.join(self.dataset_path, self.language)):
            raise ValueError("Language not found in dataset")
        if not os.path.exists(
            os.path.join(self.dataset_path, self.language, self.mls_split)
        ):
            raise ValueError("Split not found in dataset")

        # checks if the transcripts.txt file exists
        if not os.path.exists(
            os.path.join(dataset_path, language, split, "transcripts.txt")
        ):
            raise ValueError("transcripts.txt not found in dataset")

    def __get_len__(self):
        """Returns the length of the dataset"""
        return len(self.dataset_lookup)

    def __getitem__(self, idx: int) -> Sample:
        """One sample"""
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

        waveform, sample_rate = torchaudio.load(audio_path)  # type: ignore

        return Sample(
            waveform=waveform,
            spectrogram=torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=128
            )(waveform),
            utterance=utterance,
            sample_rate=sample_rate,
            speaker_id=self.dataset_lookup[idx]["speakerid"],
            book_id=self.dataset_lookup[idx]["bookid"],
            chapter_id=self.dataset_lookup[idx]["chapterid"],
        )

    def download(self, dataset_path: str, language: str):
        """Download the dataset"""
        os.makedirs(dataset_path)
        url = f"https://dl.fbaipublicfiles.com/mls/{language}.tar.gz"

        torch.hub.download_url_to_file(url, dataset_path)


class DataProcessor:
    """Factory for DataProcessingclass

    Transforms the dataset into spectrograms and labels, as well as a tokenizer
    """

    def __init__(
        self,
        dataset: MultilingualLibriSpeech,
        tokenizer_path: str,
        data_type: str = "train",
        tokenizer_type: str = "BPE",
    ):
        self.dataset = dataset
        self.data_type = data_type

        self.train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
            torchaudio.transforms.TimeMasking(time_mask_param=100),
        )

        self.valid_audio_transforms = torchaudio.transforms.MelSpectrogram()
        self.tokenizer = tokenizer_factory(
            tokenizer_path=tokenizer_path, tokenizer_type=tokenizer_type
        )

    def __call__(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Returns spectrograms, labels and their lenghts"""
        for sample in self.dataset:
            if self.data_type == "train":
                spec = (
                    self.train_audio_transforms(sample["waveform"])
                    .squeeze(0)
                    .transpose(0, 1)
                )
            elif self.data_type == "valid":
                spec = (
                    self.valid_audio_transforms(sample["waveform"])
                    .squeeze(0)
                    .transpose(0, 1)
                )
            else:
                raise ValueError("data_type should be train or valid")
            label = torch.Tensor(text_transform.encode(sample["utterance"]).ids)

            spectrograms = (
                nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
                .unsqueeze(1)
                .transpose(2, 3)
            )
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

            yield spec, label, spec.shape[0] // 2, len(labels)


if __name__ == "__main__":
    dataset_path = "/Volumes/pherkel/SWR2-ASR"
    language = "mls_german_opus"
    split = Split.train
    download = False

    dataset = MLSDataset(dataset_path, language, split, download)
    print(dataset[0])
