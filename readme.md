# SWR2-ASR

Automatic speech recognition model for the seminar "Spoken Word
Recogniton 2 (SWR2)" by Konstantin Sering in the summer term 2023.

Authors:
Silja Kasper, Marvin Borner, Philipp Merkel, Valentin Schmidt 

# Dataset
We use the german [multilangual librispeech dataset](http://www.openslr.org/94/) (mls_german_opus). If the dataset is not found under the specified path, it will be downloaded automatically.

If you want to train this model on custom data, this code expects a folder structure like this:
```
<dataset_path>
  ├── <language>
  │  ├── train
  │  │  ├── transcripts.txt
  │  │  └── audio
  │  │     └── <speakerid>
  │  │        └── <bookid>
  │  │           └── <speakerid>_<bookid>_<chapterid>.opus/.flac
  │  ├── dev
  │  │  ├── transcripts.txt
  │  │  └── audio
  │  │     └── <speakerid>
  │  │        └── <bookid>
  │  │           └── <speakerid>_<bookid>_<chapterid>.opus/.flac
  │  └── test
  │     ├── transcripts.txt
  │     └── audio
  │        └── <speakerid>
  │           └── <bookid>
  │              └── <speakerid>_<bookid>_<chapterid>.opus/.flac
``````


# Installation
The preferred method of installation is using [`poetry`](https://python-poetry.org/docs/#installation). After installing poetry, run
```
poetry install
```
to install all dependencies. `poetry` also enables you to run our scripts using
```
poetry run SCRIPT_NAME
```

Alternatively, you can use the provided `requirements.txt` file to install the dependencies using `pip` or `conda`.

# Usage

## Tokenizer

We include a pre-trained character-level tokenizer for the german language in the `data/tokenizers` directory.

If the path to the tokenizer you specified in the `config.yaml` file does not exist or is None (~), a new tokenizer will be trained on the training data.

## Decoder :
There are two options for the decoder:
- greedy
- beam search with language model

The language model is a KenLM model and supplied by the multi-lingual librispeech dataset. If you want to use a different KenLM language model, you can specify the path to the language model in the `config.yaml` file.

## Training the model

All hyperparameters can be configured in the `config.yaml` file. The main sections are:
- model
- training
- dataset
- tokenizer
- checkpoints
- inference

Train using the provided train script:

    poetry run train \
    --config_path="PATH_TO_CONFIG_FILE"

## Inference
The `config.yaml` also includes a section for inference. 
To run inference on a single audio file, run:

    poetry run recognize \
    --config_path="PATH_TO_CONFIG_FILE" \
    --file_path="PATH_TO_AUDIO_FILE" \
    --target_path="PATH_TO_TARGET_FILE"

Target path is optional. If not specified, the recognized text will be printed to the console. Otherwise, a wer will be computed.
