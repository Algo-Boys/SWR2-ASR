"""Training script for the ASR model."""
import click
import torch
import torch.nn.functional as F
import torchaudio
import yaml

from swr2_asr.model_deep_speech import SpeechRecognitionModel
from swr2_asr.utils.tokenizer import CharTokenizer


def greedy_decoder(output, tokenizer: CharTokenizer, collapse_repeated=True):
    """Greedily decode a sequence."""
    arg_maxes = torch.argmax(output, dim=2)  # pylint: disable=no-member
    blank_label = tokenizer.get_blank_token()
    decodes = []
    for args in arg_maxes:
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(tokenizer.decode(decode))
    return decodes


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
def main(config_path: str, file_path: str) -> None:
    """inference function."""
    with open(config_path, "r", encoding="utf-8") as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    # Create separate dictionaries for each top-level key
    model_config = config_dict.get("model", {})
    tokenizer_config = config_dict.get("tokenizer", {})
    inference_config = config_dict.get("inference", {})

    if inference_config["device"] == "cpu":
        device = "cpu"
    elif inference_config["device"] == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
    print(checkpoint["model_state_dict"].keys())
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
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
    output = model(spec)  # pylint: disable=not-callable
    output = F.log_softmax(output, dim=2)  # (batch, time, n_class)
    decoded_preds = greedy_decoder(output, tokenizer)

    print(decoded_preds)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
