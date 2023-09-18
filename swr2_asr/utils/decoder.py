"""Decoder for CTC-based ASR.""" ""
import os

import torch
from torchaudio.datasets.utils import _extract_tar
from torchaudio.models.decoder import ctc_decoder

from swr2_asr.utils.data import create_lexicon
from swr2_asr.utils.tokenizer import CharTokenizer


# TODO: refactor to use torch CTC decoder class
def greedy_decoder(
    output, labels, label_lengths, tokenizer: CharTokenizer, collapse_repeated=True
):  # pylint: disable=redefined-outer-name
    """Greedily decode a sequence."""
    blank_label = tokenizer.get_blank_token()
    arg_maxes = torch.argmax(output, dim=2)  # pylint: disable=no-member
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(tokenizer.decode(labels[i][: label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(tokenizer.decode(decode))
    return decodes, targets


def beam_search_decoder(
    tokenizer: CharTokenizer,  # pylint: disable=redefined-outer-name
    tokens_path: str,
    lang_model_path: str,
    language: str,
    hparams: dict,  # pylint: disable=redefined-outer-name
):
    """Beam search decoder."""

    n_gram, beam_size, beam_threshold, n_best, lm_weight, word_score = (
        hparams["n_gram"],
        hparams["beam_size"],
        hparams["beam_threshold"],
        hparams["n_best"],
        hparams["lm_weight"],
        hparams["word_score"],
    )

    if not os.path.isdir(os.path.join(lang_model_path, f"mls_lm_{language}")):
        url = f"https://dl.fbaipublicfiles.com/mls/mls_lm_{language}.tar.gz"
        torch.hub.download_url_to_file(url, f"data/mls_lm_{language}.tar.gz")
        _extract_tar("data/mls_lm_{language}.tar.gz", overwrite=True)

    if not os.path.isfile(tokens_path):
        tokenizer.create_tokens_txt(tokens_path)

    lexicon_path = os.path.join(lang_model_path, f"mls_lm_{language}", "lexicon.txt")
    if not os.path.isfile(lexicon_path):
        occurences_path = os.path.join(lang_model_path, f"mls_lm_{language}", "vocab_counts.txt")
        create_lexicon(occurences_path, lexicon_path)

    lm_path = os.path.join(lang_model_path, f"mls_lm_{language}", f"{n_gram}-gram_lm.arpa")

    decoder = ctc_decoder(
        lexicon=lexicon_path,
        tokens=tokens_path,
        lm=lm_path,
        blank_token="_",
        sil_token="<SPACE>",
        unk_word="<UNK>",
        nbest=n_best,
        beam_size=beam_size,
        beam_threshold=beam_threshold,
        lm_weight=lm_weight,
        word_score=word_score,
    )
    return decoder


if __name__ == "__main__":
    tokenizer = CharTokenizer.from_file("data/tokenizers/char_tokenizer_german.json")
    tokenizer.create_tokens_txt("data/tokenizers/tokens_german.txt")

    hparams = {
        "n_gram": 3,
        "beam_size": 100,
        "beam_threshold": 100,
        "n_best": 1,
        "lm_weight": 0.5,
        "word_score": 1.0,
    }

    beam_search_decoder(
        tokenizer,
        "data/tokenizers/tokens_german.txt",
        "data",
        "german",
        hparams,
    )
