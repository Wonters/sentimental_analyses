import torch
import torch.nn as nn

class LSTMTorchNN(nn.Module):
    do_embedding = False
    def __init__(self, vocab_size= 5000, embedding_dim=39,
                 hidden_dim = 128,
                 output_dim = 2,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.5):
        super().__init__()
        if self.do_embedding:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, **kwargs):
        if self.do_embedding:
            input_ids = self.dropout(self.embeddings(input_ids))
        if torch.backends.mps.is_available():
            input_ids =input_ids.to(torch.float32)
        # In input, shape: (batch_size, embedding_dim)
        # Needed for LSTM (batch_size, seq_len, embedding_dim)
        input_ids = input_ids.unsqueeze(1)
        lstm_out, (hidden, cell) = self.lstm(input_ids)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :,:], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        out = self.dropout(hidden)
        out = self.fc(out)
        return out

