import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TinyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.relu(self.linear(x))
        x = self.output(x)
        return x

def main():
    model = TinyModel(100, 10, 10)
    print("Tiny model defined.")

if __name__ == "__main__":
    main()
