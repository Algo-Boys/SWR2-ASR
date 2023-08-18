"""Training script for the ASR model."""
from AudioLoader.speech.mls import MultilingualLibriSpeech


def main() -> None:
    """Main function."""
    dataset = MultilingualLibriSpeech(
        "data", "mls_polish_opus", split="train", download=True
    )

    print(dataset[1])


if __name__ == "__main__":
    main()
