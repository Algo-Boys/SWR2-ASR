"""Decoder for CTC-based ASR.""" ""
import os
from dataclasses import dataclass

import torch
from torchaudio.datasets.utils import _extract_tar
from torchaudio.models.decoder import ctc_decoder

from swr2_asr.utils.data import create_lexicon
from swr2_asr.utils.tokenizer import CharTokenizer


@dataclass
class DecoderOutput:
    """Decoder output."""

    words: list[str]


def decoder_factory(decoder_type: str = "greedy") -> callable:
    """Decoder factory."""
    if decoder_type == "greedy":
        return get_greedy_decoder
    if decoder_type == "lm":
        return get_beam_search_decoder
    raise NotImplementedError


def get_greedy_decoder(
    tokenizer: CharTokenizer,  # pylint: disable=redefined-outer-name
    *_,
):
    """Greedy decoder."""
    return GreedyDecoder(tokenizer)


def get_beam_search_decoder(
    tokenizer: CharTokenizer,  # pylint: disable=redefined-outer-name
    hparams: dict,  # pylint: disable=redefined-outer-name
):
    """Beam search decoder."""
    hparams = hparams.get("lm", {})
    language, lang_model_path, n_gram, beam_size, beam_threshold, n_best, lm_weight, word_score = (
        hparams["language"],
        hparams["language_model_path"],
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

    tokens_path = os.path.join(lang_model_path, f"mls_lm_{language}", "tokens.txt")
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


class GreedyDecoder:
    """Greedy decoder."""

    def __init__(self, tokenizer: CharTokenizer):  # pylint: disable=redefined-outer-name
        self.tokenizer = tokenizer

    def __call__(
        self, output, greedy_type: str = "inference", labels=None, label_lengths=None
    ):  # pylint: disable=redefined-outer-name
        """Greedily decode a sequence."""
        if greedy_type == "train":
            res = self.train(output, labels, label_lengths)
        if greedy_type == "inference":
            res = self.inference(output)

        res = [x.split(" ") for x in res]
        res = [[DecoderOutput(x)] for x in res]
        return res

    def train(self, output, labels, label_lengths):
        """Greedily decode a sequence with known labels."""
        blank_label = tokenizer.get_blank_token()
        arg_maxes = torch.argmax(output, dim=2)  # pylint: disable=no-member
        decodes = []
        targets = []
        for i, args in enumerate(arg_maxes):
            decode = []
            targets.append(self.tokenizer.decode(labels[i][: label_lengths[i]].tolist()))
            for j, index in enumerate(args):
                if index != blank_label:
                    if j != 0 and index == args[j - 1]:
                        continue
                    decode.append(index.item())
            decodes.append(self.tokenizer.decode(decode))
        return decodes, targets

    def inference(self, output):
        """Greedily decode a sequence."""
        collapse_repeated = True
        arg_maxes = torch.argmax(output, dim=2)  # pylint: disable=no-member
        blank_label = self.tokenizer.get_blank_token()
        decodes = []
        for args in arg_maxes:
            decode = []
            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    decode.append(index.item())
            decodes.append(self.tokenizer.decode(decode))

        return decodes


if __name__ == "__main__":
    tokenizer = CharTokenizer.from_file("data/tokenizers/char_tokenizer_german.json")
    tokenizer.create_tokens_txt("data/tokenizers/tokens_german.txt")

    hparams = {
        "language": "german",
        "lang_model_path": "data",
        "n_gram": 3,
        "beam_size": 100,
        "beam_threshold": 100,
        "n_best": 1,
        "lm_weight": 0.5,
        "word_score": 1.0,
    }

    get_beam_search_decoder(tokenizer, hparams)
