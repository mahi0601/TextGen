import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextGenerator(nn.Module):
    def __init__(self, hidden_dim, num_layers, pre_trained_model="gpt2"):
        super(TextGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Load a pre-trained language model (e.g., GPT-2) from Hugging Face Transformers
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
        self.pretrained_model = AutoModel.from_pretrained(pre_trained_model)

        # Define Bi-LSTM and LSTM layers
        self.lstm = nn.LSTM(input_size=self.pretrained_model.config.hidden_size, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=True)
        
        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, self.pretrained_model.config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            embeddings = self.pretrained_model(input_ids, attention_mask=attention_mask).last_hidden_state

        lstm_out, _ = self.lstm(embeddings)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attn_output = torch.bmm(attn_weights.transpose(1, 2), lstm_out)

        logits = self.fc(attn_output.squeeze(1))
        return logits
