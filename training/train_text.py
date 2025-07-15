import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

def prepare_data():
    print("Preparing data...")
    text = "This is a tiny dataset for text generation."

    # Create a vocabulary
    vocab = sorted(list(set(text)))
    char_to_int = {c: i for i, c in enumerate(vocab)}

    # Create the dataset
    data = [char_to_int[c] for c in text]
    dataset = TensorDataset(torch.tensor(data))
    dataloader = DataLoader(dataset, batch_size=1)

    print("Data preparation complete.")
    return dataloader, len(vocab)

def main():
    dataloader, vocab_size = prepare_data()
    model = TinyModel(vocab_size, 10, 10)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(1):
        print(f"Epoch {epoch+1}")
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch[0])
            loss = nn.functional.cross_entropy(outputs, batch[0])
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss.item()}")

    print("Training complete.")

    # Save the model
    torch.save(model.state_dict(), "neuroforge_text.pt")

if __name__ == "__main__":
    main()
