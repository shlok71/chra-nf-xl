import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# This is a placeholder for the ICDAR dataset
class ICDAR_Dataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 1, 256, 256)
        self.labels = torch.randint(0, 10, (100,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 5-layer MLP model
class NCA_MLP(nn.Module):
    def __init__(self):
        super(NCA_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.mlp(x)

def main():
    dataset = ICDAR_Dataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = NCA_MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for data, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data) # Unsupervised for now
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()
