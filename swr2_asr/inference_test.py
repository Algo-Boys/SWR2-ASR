"""Training script for the ASR model."""
import torch
import torchaudio
import torchaudio.functional as F


class GreedyCTCDecoder(torch.nn.Module):
    """Greedy CTC decoder for the wav2vec2 model."""

    def __init__(self, labels, blank=0) -> None:
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


'''
Sorry marvin, Please fix this to use the new dataset
def main() -> None:
    """Main function."""
    # choose between cuda, cpu and mps devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "mps"
    device = torch.device(device)

    torch.random.manual_seed(42)

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

    print(f"Sample rate (model): {bundle.sample_rate}")
    print(f"Labels (model): {bundle.get_labels()}")

    model = bundle.get_model().to(device)

    print(model.__class__)

    # only do all things for one single sample
    dataset = MultilingualLibriSpeech("data", "mls_german_opus", split="train", download=True)

    print(dataset[0])

    # load waveforms and sample rate from dataset
    waveform, sample_rate = dataset[0]["waveform"], dataset[0]["sample_rate"]

    if sample_rate != bundle.sample_rate:
        waveform = F.resample(waveform, sample_rate, int(bundle.sample_rate))

    waveform.to(device)

    with torch.inference_mode():
        features, _ = model.extract_features(waveform)

    with torch.inference_mode():
        emission, _ = model(waveform)

    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(emission[0])

    print(transcript)
'''

if __name__ == "__main__":
    main()
