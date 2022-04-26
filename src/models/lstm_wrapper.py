import torch.nn as nn


class LSTMWrapper(nn.LSTM):
    def forward(self, x):
        return super().forward(x)[0]
