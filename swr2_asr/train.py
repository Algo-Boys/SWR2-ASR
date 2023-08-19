"""Training script for the ASR model."""
from AudioLoader.speech import MultilingualLibriSpeech
import click
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
from .loss_scores import cer, wer

MODEL_SAVE_PATH = "models/model.pt"
LOSS

class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        ä 28
        ö 29
        ü 30
        ß 31
        - 32
        é 33
        è 34
        à 35
        ù 36
        ç 37
        â 38 
        ê 39
        î 40
        ô 41
        û 42
        ë 43
        ï 44
        ü 45
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split("\n"):
            char, index = line.split()
            self.char_map[char] = int(index)
            self.index_map[int(index)] = char
        self.index_map[1] = " "

    def text_to_int(self, text):
        """Use a character map and convert text to an integer sequence"""
        int_sequence = []
        for char in text:
            if char == " ":
                mapped_char = self.char_map["<SPACE>"]
            else:
                mapped_char = self.char_map[char]
            int_sequence.append(mapped_char)
        return int_sequence

    def int_to_text(self, labels):
        """Use a character map and convert integer labels to an text sequence"""
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return "".join(string).replace("<SPACE>", " ")


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100),
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()


def data_processing(data, data_type="train"):
    """Return the spectrograms, labels, and their lengths."""
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for sample in data:
        if data_type == "train":
            spec = train_audio_transforms(sample["waveform"]).squeeze(0).transpose(0, 1)
        elif data_type == "valid":
            spec = valid_audio_transforms(sample["waveform"]).squeeze(0).transpose(0, 1)
        else:
            raise ValueError("data_type should be train or valid")
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(sample["utterance"].lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = (
        nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        .unsqueeze(1)
        .transpose(2, 3)
    )
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def greedy_decoder(
    output, labels, label_lengths, blank_label=28, collapse_repeated=True
):
    """Greedily decode a sequence."""
    arg_maxes = torch.argmax(output, dim=2)  # pylint: disable=no-member
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(
            text_transform.int_to_text(labels[i][: label_lengths[i]].tolist())
        )
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats: int):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, data):
        """x (batch, channel, feature, time)"""
        data = data.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        data = self.layer_norm(data)
        return data.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        dropout: float,
        n_feats: int,
    ):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(
            in_channels, out_channels, kernel, stride, padding=kernel // 2
        )
        self.cnn2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel,
            stride,
            padding=kernel // 2,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, data):
        """x (batch, channel, feature, time)"""
        residual = data  # (batch, channel, feature, time)
        data = self.layer_norm1(data)
        data = F.gelu(data)
        data = self.dropout1(data)
        data = self.cnn1(data)
        data = self.layer_norm2(data)
        data = F.gelu(data)
        data = self.dropout2(data)
        data = self.cnn2(data)
        data += residual
        return data  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):
    """BIdirectional GRU with Layer Normalization and Dropout"""

    def __init__(
        self,
        rnn_dim: int,
        hidden_size: int,
        dropout: float,
        batch_first: bool,
    ):
        super(BidirectionalGRU, self).__init__()

        self.bi_gru = nn.GRU(
            input_size=rnn_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """data (batch, time, feature)"""
        data = self.layer_norm(data)
        data = F.gelu(data)
        data = self.dropout(data)
        data, _ = self.bi_gru(data)
        return data


class SpeechRecognitionModel(nn.Module):
    """Speech Recognition Model Inspired by DeepSpeech 2"""

    def __init__(
        self,
        n_cnn_layers: int,
        n_rnn_layers: int,
        rnn_dim: int,
        n_class: int,
        n_feats: int,
        stride: int = 2,
        dropout: float = 0.1,
    ):
        super(SpeechRecognitionModel, self).__init__()
        n_feats //= 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)
        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(
            *[
                ResidualCNN(
                    32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats
                )
                for _ in range(n_cnn_layers)
            ]
        )
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
        self.birnn_layers = nn.Sequential(
            *[
                BidirectionalGRU(
                    rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                    hidden_size=rnn_dim,
                    dropout=dropout,
                    batch_first=i == 0,
                )
                for i in range(n_rnn_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class),
        )

    def forward(self, data):
        """data (batch, channel, feature, time)"""
        data = self.cnn(data)
        data = self.rescnn_layers(data)
        sizes = data.size()
        data = data.view(
            sizes[0], sizes[1] * sizes[2], sizes[3]
        )  # (batch, feature, time)
        data = data.transpose(1, 2)  # (batch, time, feature)
        data = self.fully_connected(data)
        data = self.birnn_layers(data)
        data = self.classifier(data)
        return data


class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):
        self.val = 0

    def step(self):
        """step"""
        self.val += 1

    def get(self):
        """get"""
        return self.val


def train(
    model,
    device,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    epoch,
    iter_meter,
):
    """Train"""
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()
        iter_meter.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print(
                f"Train Epoch: \
                    {epoch} \
                    [{batch_idx * len(spectrograms)}/{data_len} \
                    ({100.0 * batch_idx / len(train_loader)}%)]\t \
                    Loss: {loss.item()}"
            )
        return loss.item()

def test(model, device, test_loader, criterion):
    """Test"""
    print("\nevaluating...")
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for _data in test_loader:
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = greedy_decoder(
                output.transpose(0, 1), labels, label_lengths
            )
            for j, pred in enumerate(decoded_preds):
                test_cer.append(cer(decoded_targets[j], pred))
                test_wer.append(wer(decoded_targets[j], pred))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    print(
        f"Test set: Average loss:\
            {test_loss}, Average CER: {avg_cer} Average WER: {avg_wer}\n"
    )


def run(learning_rate: float = 5e-4, batch_size: int = 8, epochs: int = 3) -> None:
    """Runs the training script."""
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 46,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
    }

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")  # pylint: disable=no-member
    # device = torch.device("mps")

    train_dataset = MultilingualLibriSpeech(
        "/Volumes/pherkel/SWR2-ASR/", "mls_german_opus", split="dev", download=False
    )
    test_dataset = MultilingualLibriSpeech(
        "/Volumes/pherkel/SWR2-ASR/", "mls_german_opus", split="test", download=False
    )

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        collate_fn=lambda x: data_processing(x, "train"),
        **kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        collate_fn=lambda x: data_processing(x, "train"),
        **kwargs,
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

    print(
        "Num Model Parameters", sum([param.nelement() for param in model.parameters()])
    )

    optimizer = optim.AdamW(model.parameters(), hparams["learning_rate"])
    criterion = nn.CTCLoss(blank=28).to(device)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams["learning_rate"],
        steps_per_epoch=int(len(train_loader)),
        epochs=hparams["epochs"],
        anneal_strategy="linear",
    )

    iter_meter = IterMeter()
    for epoch in range(1, epochs + 1):
        loss = train(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            epoch,
            iter_meter,
        )
        if epoch%3 == 0 or epoch == epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss},MODEL_SAVE_PATH)
        test(model=model, device=device, test_loader=test_loader, criterion=criterion)


@click.command()
@click.option("--learning-rate", default=1e-3, help="Learning rate")
@click.option("--batch_size", default=1, help="Batch size")
@click.option("--epochs", default=1, help="Number of epochs")
def run_cli(learning_rate: float, batch_size: int, epochs: int) -> None:
    """Runs the training script."""
    run(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)


if __name__ == "__main__":
    run(learning_rate=5e-4, batch_size=16, epochs=1)
