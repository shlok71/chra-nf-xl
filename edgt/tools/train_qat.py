import torch
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert
from edgt.model import get_model

def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    for i, (images, target) in enumerate(data_loader):
        images, target = images.to(device), target.to(device)
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    # --- Model ---
    model = get_model()
    model.train()

    # --- QAT ---
    model.fuse_model()
    model.qconfig = get_default_qat_qconfig('fbgemm')
    prepare_qat(model, inplace=True)

    # --- Dataloader ---
    # Using a dummy dataloader for demonstration purposes
    dataset = torch.randn(100, 3, 224, 224)
    labels = torch.randint(0, 1000, (100,))
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dataset, labels),
        batch_size=10
    )

    # --- Training ---
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    device = torch.device('cpu')
    train_one_epoch(model, criterion, optimizer, data_loader, device)

    # --- Convert to quantized model ---
    model.eval()
    model_quantized = convert(model, inplace=False)

    # --- Save quantized model ---
    torch.save(model_quantized.state_dict(), "edgt/model_quantized.pth")

if __name__ == '__main__':
    main()
