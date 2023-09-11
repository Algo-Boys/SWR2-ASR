"""Training script for the ASR model."""
from typing import TypedDict

import torch
import torch.nn.functional as F
import torchaudio

from swr2_asr.model_deep_speech import SpeechRecognitionModel
from swr2_asr.utils.tokenizer import CharTokenizer


class HParams(TypedDict):
    """Type for the hyperparameters of the model."""

    n_cnn_layers: int
    n_rnn_layers: int
    rnn_dim: int
    n_class: int
    n_feats: int
    stride: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int


def greedy_decoder(output, tokenizer, collapse_repeated=True):
    """Greedily decode a sequence."""
    arg_maxes = torch.argmax(output, dim=2)  # pylint: disable=no-member
    blank_label = tokenizer.encode(" ").ids[0]
    decodes = []
    for _i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(tokenizer.decode(decode))
    return decodes


def main() -> None:
    """inference function."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)  # pylint: disable=no-member

    tokenizer = CharTokenizer.from_file("char_tokenizer_german.json")

    spectrogram_hparams = {
        "sample_rate": 16000,
        "n_fft": 400,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": 128,
        "f_min": 0,
        "f_max": 8000,
        "power": 2.0,
    }

    hparams = HParams(
        n_cnn_layers=3,
        n_rnn_layers=5,
        rnn_dim=512,
        n_class=tokenizer.get_vocab_size(),
        n_feats=128,
        stride=2,
        dropout=0.1,
        learning_rate=0.1,
        batch_size=30,
        epochs=100,
    )

    model = SpeechRecognitionModel(
        hparams["n_cnn_layers"],
        hparams["n_rnn_layers"],
        hparams["rnn_dim"],
        hparams["n_class"],
        hparams["n_feats"],
        hparams["stride"],
        hparams["dropout"],
    ).to(device)

    checkpoint = torch.load("model8", map_location=device)
    state_dict = {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)

    # waveform, sample_rate = torchaudio.load("test.opus")
    waveform, sample_rate = torchaudio.load("marvin_rede.flac")  # pylint: disable=no-member
    if sample_rate != spectrogram_hparams["sample_rate"]:
        resampler = torchaudio.transforms.Resample(sample_rate, spectrogram_hparams["sample_rate"])
        waveform = resampler(waveform)

    spec = (
        torchaudio.transforms.MelSpectrogram(**spectrogram_hparams)(waveform)
        .squeeze(0)
        .transpose(0, 1)
    )
    specs = [spec]
    specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True).unsqueeze(1).transpose(2, 3)

    output = model(specs)  # pylint: disable=not-callable
    output = F.log_softmax(output, dim=2)
    output = output.transpose(0, 1)  # (time, batch, n_class)
    decodes = greedy_decoder(output, tokenizer)
    print(decodes)


if __name__ == "__main__":
    main()
