"""Main definition of the Deep speech 2 model by Baidu Research.

Following definition by Assembly AI 
(https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/)
"""
import torch.nn.functional as F
from torch import nn


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats: int):
        super().__init__()
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
        super().__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
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
    """Bidirectional GRU with Layer Normalization and Dropout"""

    def __init__(
        self,
        rnn_dim: int,
        hidden_size: int,
        dropout: float,
        batch_first: bool,
    ):
        super().__init__()

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
        super().__init__()
        n_feats //= 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)
        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(
            *[
                ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
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
        data = data.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        data = data.transpose(1, 2)  # (batch, time, feature)
        data = self.fully_connected(data)
        data = self.birnn_layers(data)
        data = self.classifier(data)
        return data
