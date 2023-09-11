"""Class containing utils for the ASR system."""
import os
from enum import Enum

import numpy as np
import torch
import torchaudio
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets.utils import _extract_tar

from swr2_asr.utils.tokenizer import CharTokenizer


class DataProcessing:
    """Data processing class for the dataloader"""

    def __init__(self, data_type: str, tokenizer: CharTokenizer):
        self.data_type = data_type
        self.tokenizer = tokenizer

        if data_type == "train":
            self.audio_transform = torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                torchaudio.transforms.TimeMasking(time_mask_param=100),
            )
        elif data_type == "valid":
            self.audio_transform = torchaudio.transforms.MelSpectrogram()

    def __call__(self, data) -> tuple[Tensor, Tensor, list, list]:
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        for waveform, _, utterance, _, _, _ in data:
            spec = self.audio_transform(waveform).squeeze(0).transpose(0, 1)
            spectrograms.append(spec)
            label = torch.Tensor(self.tokenizer.encode(utterance.lower()))
            labels.append(label)
            input_lengths.append(spec.shape[0] // 2)
            label_lengths.append(len(label))

        spectrograms = (
            nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        )
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths


# create enum specifiying dataset splits
class MLSSplit(str, Enum):
    """Enum specifying dataset as they are defined in the
    Multilingual LibriSpeech dataset"""

    TRAIN = "train"
    TEST = "test"
    DEV = "dev"


class Split(str, Enum):
    """Extending the MLSSplit class to allow for a custom validation split"""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    DEV = "dev"


def split_to_mls_split(split_name: Split) -> MLSSplit:
    """Converts the custom split to a MLSSplit"""
    if split_name == Split.VALID:
        return MLSSplit.TRAIN
    return split_name  # type: ignore


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
        split: Split,  # pylint: disable=redefined-outer-name
        limited: bool = False,
        download: bool = True,
        size: float = 0.2,
    ):
        """Initializes the dataset"""
        self.dataset_path = dataset_path
        self.language = language
        self.file_ext = ".opus" if "opus" in language else ".flac"
        self.mls_split: MLSSplit = split_to_mls_split(split)  # split path on disk
        self.split: Split = split  # split used internally

        self.dataset_lookup = []

        self._handle_download_dataset(download)
        self._validate_local_directory()
        if limited and split in (Split.TRAIN, Split.VALID):
            self.initialize_limited()
        else:
            self.initialize()

        self.dataset_lookup = self.dataset_lookup[: int(len(self.dataset_lookup) * size)]

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

    def _handle_download_dataset(self, download: bool) -> None:
        """Download the dataset"""
        if not download:
            print("Download flag not set, skipping download")
            return
        # zip exists:
        if os.path.isfile(os.path.join(self.dataset_path, self.language) + ".tar.gz") and download:
            print(f"Found dataset at {self.dataset_path}. Skipping download")
        # path exists:
        elif os.path.isdir(os.path.join(self.dataset_path, self.language)) and download:
            return
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
            _extract_tar(os.path.join(self.dataset_path, self.language) + ".tar.gz", overwrite=True)
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

    def __getitem__(self, idx: int) -> tuple[Tensor, int, str, int, int, int]:
        """One sample

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            int:
                Speaker ID
            int:
                Chapter ID
            int:
                Utterance ID
        """
        # get the utterance
        dataset_lookup_entry = self.dataset_lookup[idx]

        utterance = dataset_lookup_entry["utterance"]

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
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        return (
            waveform,
            sample_rate,
            utterance,
            dataset_lookup_entry["speakerid"],
            dataset_lookup_entry["chapterid"],
            idx,
        )  # type: ignore


if __name__ == "__main__":
    DATASET_PATH = "/Volumes/pherkel/SWR2-ASR"
    LANGUAGE = "mls_german_opus"
    split = Split.DEV
    DOWNLOAD = False

    dataset = MLSDataset(DATASET_PATH, LANGUAGE, split, download=DOWNLOAD)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=DataProcessing(
            "train", CharTokenizer.from_file("data/tokenizers/char_tokenizer_german.json")
        ),
    )

    for batch in dataloader:
        print(batch)
        break

    print(len(dataset))

    print(dataset[0])
