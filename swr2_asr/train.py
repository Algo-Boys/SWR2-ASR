# import dataset
# define model
# define loss
# define optimizer
# train
from AudioLoader.speech.mls import MultilingualLibriSpeech


def main():
    dataset = MultilingualLibriSpeech(
        "data", "mls_polish_opus", split="train", download=True
    )

    print(dataset[1])


if __name__ == "__main__":
    main()
