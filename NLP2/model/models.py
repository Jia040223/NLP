import torch
from torch import nn

class FNN(nn.Module):
    def __init__(self, vocab_size, seq_len, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.flatten = nn.Flatten()

        input_size = (seq_len - 1) * embedding_dim
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU()])
            input_size = hidden_size

        layers.extend([
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU()])

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = torch.matmul(x, self.embedding.weight.T)

        return x


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.rnns = nn.ModuleList([nn.RNN(embedding_dim if layer == 0 else hidden_size,
                                          hidden_size, batch_first=True) for layer in range(num_layers)])
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)

        for rnn in self.rnns:
            x, _ = rnn(x)

        x = self.linear_relu_stack(x)
        x = torch.matmul(x, self.embedding.weight.T)

        return x
    
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.lstms = nn.ModuleList([nn.LSTM(embedding_dim if layer == 0 else hidden_size,
                                            hidden_size, batch_first=True) for layer in range(num_layers)])
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)

        for lstm in self.lstms:
            x, _ = lstm(x)

        x = self.linear_relu_stack(x)
        x = torch.matmul(x, self.embedding.weight.T)

        return x
