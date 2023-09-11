"""Tokenizer for Multilingual Librispeech datasets"""


class CharTokenizer:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        _
        <BLANK>
        <UNK>
        <SPACE>
        a
        b
        c
        d
        e
        f
        g
        h
        i
        j
        k
        l
        m
        n
        o
        p
        q
        r
        s
        t
        u
        v
        w
        x
        y
        z
        é
        à
        ä
        ö
        ß
        ü
        -
        '
        """

        self.char_map = {}
        self.index_map = {}
        for idx, char in enumerate(char_map_str.strip().split("\n")):
            char = char.strip()
            self.char_map[char] = idx
            self.index_map[idx] = char
        self.index_map[1] = " "

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

    # TODO: add train function

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
                    tokenizer.char_map[char] = int(index)
                    tokenizer.index_map[int(index)] = char
        return load_tokenizer


if __name__ == "__main__":
    tokenizer = CharTokenizer()
    tokenizer.save("data/tokenizers/char_tokenizer_german.json")
    print(tokenizer.char_map)
    print(tokenizer.index_map)
    print(tokenizer.get_vocab_size())
    print(tokenizer.get_blank_token())
    print(tokenizer.get_unk_token())
    print(tokenizer.get_space_token())
    print(tokenizer.encode("hallo welt"))
    print(tokenizer.decode([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
