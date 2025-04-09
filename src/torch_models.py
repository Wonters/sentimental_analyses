import torch
import torch.nn as nn
from transformers import AutoModel

class LSTMTorchNN(nn.Module):
    def __init__(self, vocab_size= 5000, embedding_dim=39,
                 hidden_dim = 128,
                 output_dim = 2,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.5):
        super().__init__()
        self.bert_embeddings = AutoModel.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
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
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids, **kwargs):
        embeddings = self.bert_embeddings(input_ids=input_ids).last_hidden_state
        input_ids = self.dropout(embeddings)
        #input_ids = self.dropout(self.embeddings(input_ids))
        lstm_out, (hidden, cell) = self.lstm(input_ids)
 
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :,:], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        out = self.classifier(hidden)
        return out.squeeze(-1)

