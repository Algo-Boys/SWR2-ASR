"""Tokenizer for use with Multilingual Librispeech"""
from dataclasses import dataclass
import json
import os
import click
from tqdm import tqdm

from AudioLoader.speech import MultilingualLibriSpeech

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


@dataclass
class Encoding:
    """Simple dataclass to represent an encoding"""

    ids: list[int]


class CharTokenizer:
    """Very simple tokenizer for use with Multilingual Librispeech

    Simply checks what characters are in the dataset and uses them as tokens.

    Exposes the same interface as tokenizers from the huggingface library, i.e.
    encode, decode, decode_batch, save, from_file and train.
    """

    def __init__(self):
        self.char_map = {}
        self.index_map = {}
        self.add_tokens(["<UNK>", "<SPACE>"])

    def add_tokens(self, tokens: list[str]):
        """Manually add tokens to the tokenizer

        Args:
            tokens (list[str]): List of tokens to add
        """
        for token in tokens:
            if token not in self.char_map:
                self.char_map[token] = len(self.char_map)
                self.index_map[len(self.index_map)] = token

    def train(
        self, dataset_path: str, language: str, split: str, download: bool = True
    ):
        """Train the tokenizer on the given dataset

        Args:
            dataset_path (str): Path to the MLS dataset
            language (str): Language to use
            split (str): Split to use
        """
        if split not in ["train", "dev", "test", "all"]:
            raise ValueError("Split must be one of train, dev, test, all")

        if split == "all":
            splits = ["train", "dev", "test"]
        else:
            splits = [split]

        chars = set()
        for sp in splits:
            transcript_path = os.path.join(
                dataset_path, language, sp, "transcripts.txt"
            )

            # check if dataset is downloaded, download if not
            if download and not os.path.exists(transcript_path):
                MultilingualLibriSpeech(dataset_path, language, sp, download=True)

            with open(
                transcript_path,
                "r",
                encoding="utf-8",
            ) as file:
                lines = file.readlines()
            lines = [line.split(" ", 1)[1] for line in lines]
            lines = [line.strip() for line in lines]

            for line in tqdm(lines, desc=f"Training tokenizer on {sp} split"):
                chars.update(line)
        offset = len(self.char_map)
        for i, char in enumerate(chars):
            i += offset
            self.char_map[char] = i
            self.index_map[i] = char

    def encode(self, text: str):
        """Use a character map and convert text to an integer sequence

        automatically maps spaces to <SPACE> and makes everything lowercase
        unknown characters are mapped to the <UNK> token

        """
        int_sequence = []
        text = text.lower()
        for char in text:
            if char == " ":
                mapped_char = self.char_map["<SPACE>"]
            elif char not in self.char_map:
                mapped_char = self.char_map["<UNK>"]
            else:
                mapped_char = self.char_map[char]
            int_sequence.append(mapped_char)
        return Encoding(ids=int_sequence)

    def decode(self, labels: list[int], remove_special_tokens: bool = True):
        """Use a character map and convert integer labels to an text sequence

        Args:
            labels (list[int]): List of integer labels
            remove_special_tokens (bool): Whether to remove special tokens.
                Defaults to True.
        """
        string = []
        for i in labels:
            if remove_special_tokens and self.index_map[f"{i}"] == "<UNK>":
                continue
            if remove_special_tokens and self.index_map[f"{i}"] == "<SPACE>":
                string.append(" ")
            string.append(self.index_map[f"{i}"])
        return "".join(string).replace("<SPACE>", " ")

    def decode_batch(self, labels: list[list[int]]):
        """Use a character map and convert integer labels to an text sequence"""
        strings = []
        for label in labels:
            string = []
            for i in label:
                if self.index_map[i] == "<UNK>":
                    continue
                if self.index_map[i] == "<SPACE>":
                    string.append(" ")
                string.append(self.index_map[i])
            strings.append("".join(string).replace("<SPACE>", " "))
        return strings

    def save(self, path: str):
        """Save the tokenizer to a file"""
        with open(path, "w", encoding="utf-8") as file:
            # save it in the following format:
            # {"char_map": {"a": 0, "b": 1, ...}, "index_map": {0: "a", 1: "b", ...}}
            json.dump(
                {"char_map": self.char_map, "index_map": self.index_map},
                file,
                ensure_ascii=False,
            )

    def from_file(self, path: str):
        """Load the tokenizer from a file"""
        with open(path, "r", encoding="utf-8") as file:
            # load it in the following format:
            # {"char_map": {"a": 0, "b": 1, ...}, "index_map": {0: "a", 1: "b", ...}}
            saved_file = json.load(file)
        self.char_map = saved_file["char_map"]
        self.index_map = saved_file["index_map"]


@click.command()
@click.option("--dataset_path", default="data", help="Path to the MLS dataset")
@click.option("--language", default="mls_german_opus", help="Language to use")
@click.option("--split", default="train", help="Split to use (including all)")
@click.option("--download", default=True, help="Whether to download the dataset")
@click.option(
    "--out_path", default="tokenizer.json", help="Path to save the tokenizer to"
)
@click.option("--vocab_size", default=2000, help="Size of the vocabulary")
def train_bpe_tokenizer(
    dataset_path: str,
    language: str,
    split: str,
    out_path: str,
    download: bool,
    vocab_size: int,
):
    """Train a Byte-Pair Encoder tokenizer on the MLS dataset

    Assumes that the MLS dataset is located in the dataset_path and there is a
    transcripts.txt file in the split folder.

    Args:
        dataset_path (str): Path to the MLS dataset
        language (str): Language to use
        split (str): Split to use
        download (bool): Whether to download the dataset if it is not present
        out_path (str): Path to save the tokenizer to
        vocab_size (int): Size of the vocabulary
    """
    if split not in ["train", "dev", "test", "all"]:
        raise ValueError("Split must be one of train, dev, test, all")

    if split == "all":
        splits = ["train", "dev", "test"]
    else:
        splits = [split]

    lines = []

    for sp in splits:
        transcripts_path = os.path.join(dataset_path, language, sp, "transcripts.txt")
        if download and not os.path.exists(transcripts_path):
            MultilingualLibriSpeech(dataset_path, language, sp, download=True)

        with open(
            transcripts_path,
            "r",
            encoding="utf-8",
        ) as file:
            sp_lines = file.readlines()
        sp_lines = [line.split(" ", 1)[1] for line in sp_lines]
        sp_lines = [line.strip() for line in sp_lines]

        lines.append(sp_lines)

    bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    initial_alphabet = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "ä",
        "ö",
        "ü",
        "ß",
        "-",
        "é",
        "è",
        "à",
        "ù",
        "ç",
        "â",
        "ê",
        "î",
        "ô",
        "û",
        "ë",
        "ï",
        "ü",
    ]

    trainer = BpeTrainer(
        special_tokens=["[UNK]"],
        vocab_size=vocab_size,
        initial_alphabet=initial_alphabet,
        show_progress=True,
    )  # type: ignore

    bpe_tokenizer.pre_tokenizer = Whitespace()  # type: ignore

    bpe_tokenizer.normalizer = normalizers.Lowercase()  # type: ignore

    bpe_tokenizer.train_from_iterator(lines, trainer=trainer)

    bpe_tokenizer.save(out_path)


@click.command()
@click.option("--dataset_path", default="data", help="Path to the MLS dataset")
@click.option("--language", default="mls_german_opus", help="Language to use")
@click.option("--split", default="train", help="Split to use")
@click.option(
    "--out_path", default="tokenizer_chars.txt", help="Path to save the tokenizer to"
)
@click.option("--download", default=True, help="Whether to download the dataset")
def train_char_tokenizer(
    dataset_path: str,
    language: str,
    split: str,
    out_path: str,
    download: bool,
):
    """Train a Byte-Pair Encoder tokenizer on the MLS dataset

    Assumes that the MLS dataset is located in the dataset_path and there is a
    transcripts.txt file in the split folder.

    Args:
        dataset_path (str): Path to the MLS dataset
        language (str): Language to use
        split (str): Split to use
        download (bool): Whether to download the dataset if it is not present
        out_path (str): Path to save the tokenizer to
    """
    char_tokenizer = CharTokenizer()

    char_tokenizer.train(dataset_path, language, split, download)

    char_tokenizer.save(out_path)


if __name__ == "__main__":
    tokenizer = CharTokenizer()
    tokenizer.from_file("data/tokenizers/char_tokenizer_german.json")

    print(tokenizer.decode(tokenizer.encode("Fichier non trouvé").ids))
