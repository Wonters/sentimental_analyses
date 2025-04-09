import torch
import torch.nn as nn

class LSTMTorchNN(nn.Module):
    def __init__(self, vocab_size= 5000, embedding_dim=39,
                 hidden_dim = 128,
                 output_dim = 2,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.5):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, **kwargs):
        input_ids = self.dropout(self.embeddings(input_ids))
        lstm_out, (hidden, cell) = self.lstm(input_ids)
 
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :,:], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        out = self.dropout(hidden)
        out = self.fc(out)
        return out

