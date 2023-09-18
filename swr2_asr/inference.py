"""Training script for the ASR model."""
from typing import Union

import click
import torch
import torch.nn.functional as F
import torchaudio
import yaml

from swr2_asr.model_deep_speech import SpeechRecognitionModel
from swr2_asr.utils.decoder import decoder_factory
from swr2_asr.utils.loss_scores import wer
from swr2_asr.utils.tokenizer import CharTokenizer


@click.command()
@click.option(
    "--config_path",
    default="config.yaml",
    help="Path to yaml config file",
    type=click.Path(exists=True),
)
@click.option(
    "--file_path",
    help="Path to audio file",
    type=click.Path(exists=True),
)
# optional arguments
@click.option(
    "--target_path",
    help="Path to target text file",
)
def main(config_path: str, file_path: str, target_path: Union[str, None] = None) -> None:
    """inference function."""
    with open(config_path, "r", encoding="utf-8") as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    # Create separate dictionaries for each top-level key
    model_config = config_dict.get("model", {})
    tokenizer_config = config_dict.get("tokenizer", {})
    inference_config = config_dict.get("inference", {})
    decoder_config = config_dict.get("decoder", {})

    if inference_config.get("device", "") == "cpu":
        device = "cpu"
    elif inference_config.get("device", "") == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif inference_config.get("device", "") == "mps":
        device = "mps"
    else:
        device = "cpu"
    device = torch.device(device)  # pylint: disable=no-member

    tokenizer = CharTokenizer.from_file(tokenizer_config["tokenizer_path"])

    model = SpeechRecognitionModel(
        model_config["n_cnn_layers"],
        model_config["n_rnn_layers"],
        model_config["rnn_dim"],
        tokenizer.get_vocab_size(),
        model_config["n_feats"],
        model_config["stride"],
        model_config["dropout"],
    ).to(device)

    checkpoint = torch.load(inference_config["model_load_path"], map_location=device)

    state_dict = {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    waveform, sample_rate = torchaudio.load(file_path)  # pylint: disable=no-member
    if waveform.shape[0] != 1:
        waveform = waveform[1]
        waveform = waveform.unsqueeze(0)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    data_processing = torchaudio.transforms.MelSpectrogram(n_mels=model_config["n_feats"])

    spec = data_processing(waveform).squeeze(0).transpose(0, 1)

    spec = spec.unsqueeze(0)
    spec = spec.transpose(1, 2)
    spec = spec.unsqueeze(0)
    spec = spec.to(device)
    output = model(spec)  # pylint: disable=not-callable
    output = F.log_softmax(output, dim=2)  # (batch, time, n_class)

    decoder = decoder_factory(decoder_config["type"])(tokenizer, decoder_config)

    preds = decoder(output)
    preds = " ".join(preds[0][0].words).strip()

    if target_path is not None:
        with open(target_path, "r", encoding="utf-8") as target_file:
            target = target_file.read()
            target = target.lower()
            target = target.replace("«", "")
            target = target.replace("»", "")
            target = target.replace(",", "")
            target = target.replace(".", "")
            target = target.replace("?", "")
            target = target.replace("!", "")

        print("---------")
        print(f"Prediction:\n\{preds}")
        print("---------")
        print(f"Target:\n{target}")
        print("---------")
        print(f"WER: {wer(preds, target)}")

    else:
        print(f"Prediction:\n{preds}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
