"""Main entrypoint for swr2-asr."""
import torch
import torchaudio

if __name__ == "__main__":
    # test if GPU is available
    print("GPU available: ", torch.cuda.is_available())

    # test if torchaudio is installed correctly
    print("torchaudio version: ", torchaudio.__version__)
    print("torchaudio backend: ", torchaudio.get_audio_backend())
    print("torchaudio info: ", torchaudio.get_audio_backend())
