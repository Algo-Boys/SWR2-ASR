"""Training script for the ASR model."""
from AudioLoader.speech.mls import MultilingualLibriSpeech
import os


def main() -> None:
    """Main function."""
    dataset = MultilingualLibriSpeech(
        "data", "mls_polish_opus", split="train", download=(not os.path.isdir("data"))
    )

    print(dataset[1])


if __name__ == "__main__":
    main()
