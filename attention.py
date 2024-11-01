import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.bmm(weights.transpose(1, 2), lstm_output)
        return context_vector.squeeze(1)
