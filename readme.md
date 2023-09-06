# SWR2-ASR

Automatic speech recognition model for the seminar spoken word
recogniton 2 (SWR2) in the summer term 2023.

# Installation
```
poetry install
```

# Usage

## Training the tokenizer
We use a byte pair encoding tokenizer. To train the tokenizer, run
```
poetry run train-bpe-tokenizer --dataset_path="DATA_PATH" --language=mls_german_opus --split=all --out_path="data/tokenizers/bpe_tokenizer_german_3000.json" --vocab_size=3000
```
with the desired values for `DATA_PATH` and `vocab_size`.

You can also use a character level tokenizer, which can be trained with
```
poetry run train-char-tokenizer --dataset_path="DATA_PATH" --language=mls_german_opus --split=all --out_path="data/tokenizers/char_tokenizer_german.txt"
```
## Training

Train using the provided train script:

    poetry run train

## Evaluation

## Inference

    poetry run recognize

## CI

You can use the Makefile to run these commands manually

    make format

    make lint
