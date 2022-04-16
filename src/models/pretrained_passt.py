from hear21passt.base import load_model
import torch
import torch.nn as nn


class PretrainedPaSST(nn.Module):
    def __init__(self, num_new_classes: int, max_audio_len: float):
        super().__init__()
        assert num_new_classes >= 0 and max_audio_len > 0
        self.net = load_model(mode='logits').train()

        # initialize time positional embeddings
        embed_dim = self.net.net.time_new_pos_embed.shape[1]
        time_dim = self.net.net.time_new_pos_embed.shape[3]
        time_embed = nn.Parameter(
            torch.zeros(1, embed_dim, 1, int(100 * max_audio_len))
        )
        time_embed[:, :, :, :time_dim].data =\
            self.net.net.time_new_pos_embed.data
        self.net.net.time_new_pos_embed = time_embed

        # initialize classifier head
        in_f = self.net.net.head[1].in_features
        out_f = self.net.net.head[1].out_features
        new_head = nn.Linear(in_f, out_f + num_new_classes)
        new_head.weight.data[:out_f] = self.net.net.head[1].weight.data
        new_head.bias.data[:out_f] = self.net.net.head[1].bias.data
        self.net.net.head[1] = new_head

        # freeze parameters
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.net.head[1].requires_grad_(True)
        self.net.net.time_new_pos_embed.requires_grad_(True)

    def forward(self, x):
        return self.net(x)
