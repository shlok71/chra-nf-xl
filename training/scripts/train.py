import torch
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Config
from data_preparation import prepare_data

def main():
    config = GPT2Config.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel(config)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    dataloader = prepare_data()
    dataloader.batch_size = 1

    for epoch in range(1):
        print(f"Epoch {epoch+1}")
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch[0], labels=batch[0])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss.item()}")

    print("Training complete.")

if __name__ == "__main__":
    main()
