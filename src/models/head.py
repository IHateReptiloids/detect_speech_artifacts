import torch.nn as nn


class LSTMWrapper(nn.LSTM):
    def forward(self, x):
        return super().forward(x)[0]


class Head(nn.Module):
    def __init__(self, args, in_f, num_classes):
        super().__init__()

        self.head = None
        if args.head_type == 'linear':
            self.head = nn.Linear(in_f, num_classes)
        elif args.head_type == 'lstm':
            self.head = nn.Sequential(
                nn.Linear(in_f, args.hidden_size),
                nn.SiLU(),
                LSTMWrapper(
                    args.hidden_size, args.hidden_size,
                    bidirectional=args.bidirectional, batch_first=True
                ),
                nn.Linear(args.hidden_size * (1 + int(args.bidirectional)),
                          num_classes)
            )
        else:
            raise ValueError('Unknown head type')
    
    def forward(self, x):
        return self.head(x)
