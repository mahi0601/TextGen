import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)  # shape: (batch, seq, 1)
        context = torch.bmm(weights.transpose(1, 2), lstm_output)  # shape: (batch, 1, hidden*2)
        return context.squeeze(1)  # shape: (batch, hidden*2)