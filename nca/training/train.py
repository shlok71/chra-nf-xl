import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from PIL import Image
import os

class ICDAR_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".png"):
                    self.image_files.append(os.path.join(subdir, file))
                    label_file = os.path.join(subdir, "label.txt")
                    with open(label_file, 'r') as f:
                        self.labels.append(f.read().strip())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("L")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

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
    # Character set for transcription
    CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"
    char_to_int = {c: i + 1 for i, c in enumerate(CHARSET)}
    int_to_char = {i + 1: c for i, c in enumerate(CHARSET)}

    transform = transforms.Compose([
        transforms.Resize((32, 100)),
        transforms.ToTensor(),
    ])
    dataset = ICDAR_Dataset(root_dir="path/to/icdar", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = NCA_MLP()
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images).log_softmax(2)
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)

            target_labels = [torch.tensor([char_to_int[c] for c in label], dtype=torch.long) for label in labels]
            target_lengths = torch.tensor([len(label) for label in target_labels], dtype=torch.long)
            targets = torch.cat(target_labels)

            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Save the model's weights
    with open("nca_model.bin", "wb") as f:
        for param in model.parameters():
            f.write(param.data.numpy().tobytes())

if __name__ == "__main__":
    main()
