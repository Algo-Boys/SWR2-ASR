"""Tokenizer for Multilingual Librispeech datasets"""
import os
from datetime import datetime

from tqdm.autonotebook import tqdm


class CharTokenizer:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        self.char_map = {}
        self.index_map = {}

    def encode(self, text: str) -> list[int]:
        """Use a character map and convert text to an integer sequence"""
        int_sequence = []
        for char in text:
            if char == " ":
                char = self.char_map["<SPACE>"]
            elif char not in self.char_map:
                char = self.char_map["<UNK>"]
            else:
                char = self.char_map[char]
            int_sequence.append(char)
        return int_sequence

    def decode(self, labels: list[int]) -> str:
        """Use a character map and convert integer labels to an text sequence"""
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return "".join(string).replace("<SPACE>", " ")

    def get_vocab_size(self) -> int:
        """Get the number of unique characters in the dataset"""
        return len(self.char_map)

    def get_blank_token(self) -> int:
        """Get the integer representation of the <BLANK> character"""
        return self.char_map["<BLANK>"]

    def get_unk_token(self) -> int:
        """Get the integer representation of the <UNK> character"""
        return self.char_map["<UNK>"]

    def get_space_token(self) -> int:
        """Get the integer representation of the <SPACE> character"""
        return self.char_map["<SPACE>"]

    @staticmethod
    def train(dataset_path: str, language: str) -> "CharTokenizer":
        """Train the tokenizer on a dataset"""
        chars = set()
        root_path = os.path.join(dataset_path, language)
        for split in os.listdir(root_path):
            split_dir = os.path.join(root_path, split)
            if os.path.isdir(split_dir):
                transcript_path = os.path.join(split_dir, "transcripts.txt")

                with open(transcript_path, "r", encoding="utf-8") as transcrips:
                    lines = transcrips.readlines()
                lines = [line.split(" ", 1)[1] for line in lines]
                lines = [line.strip() for line in lines]
                lines = [line.lower() for line in lines]

                for line in tqdm(lines, desc=f"Training tokenizer on {split_dir} split"):
                    chars.update(line)

        # sort chars
        chars.remove(" ")
        chars = sorted(chars)

        train_tokenizer = CharTokenizer()

        train_tokenizer.char_map["_"] = 0
        train_tokenizer.char_map["<BLANK>"] = 1
        train_tokenizer.char_map["<UNK>"] = 2
        train_tokenizer.char_map["<SPACE>"] = 3

        train_tokenizer.index_map[0] = "_"
        train_tokenizer.index_map[1] = "<BLANK>"
        train_tokenizer.index_map[2] = "<UNK>"
        train_tokenizer.index_map[3] = "<SPACE>"

        offset = 4

        for idx, char in enumerate(chars):
            idx += offset
            train_tokenizer.char_map[char] = idx
            train_tokenizer.index_map[idx] = char

        train_tokenizer_dir = os.path.join("data/tokenizers")
        train_tokenizer_path = os.path.join(
            train_tokenizer_dir,
            f"char_tokenizer_{language}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json",
        )

        if not os.path.exists(os.path.dirname(train_tokenizer_dir)):
            os.makedirs(train_tokenizer_dir)
        train_tokenizer.save(train_tokenizer_path)

        return train_tokenizer

    def save(self, path: str) -> None:
        """Save the tokenizer to a file"""
        with open(path, "w", encoding="utf-8") as file:
            for char, index in self.char_map.items():
                file.write(f"{char} {index}\n")

    @staticmethod
    def from_file(tokenizer_file: str) -> "CharTokenizer":
        """Instantiate a CharTokenizer from a file"""
        load_tokenizer = CharTokenizer()
        with open(tokenizer_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    char, index = line.split()
                    load_tokenizer.char_map[char] = int(index)
                    load_tokenizer.index_map[int(index)] = char
        return load_tokenizer
    

    #TO DO check about the weird unknown tokens etc.
    def create_txt(self,path:str):
      with open(path, 'w',encoding="utf-8") as file:
        for key,value in self.char_map():
           file.write(f"{key}\n")
        