import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def prepare_data():
    print("Preparing data...")
    # This is a placeholder for loading the raw data.
    # A real implementation would load data from the specified paths.
    text_data = np.random.rand(100, 16384)
    ocr_data = np.random.rand(100, 16384)
    canvas_data = np.random.rand(100, 16384)
    web_text_data = np.random.rand(100, 16384)
    reasoning_data = np.random.rand(100, 16384)

    # This is a placeholder for creating the multi-task batches.
    # A real implementation would create a dataset that yields
    # batches of data from different tasks.
    dataset = TensorDataset(
        torch.from_numpy(np.concatenate([text_data, ocr_data, canvas_data, web_text_data, reasoning_data])).float()
    )
    dataloader = DataLoader(dataset, batch_size=10)
    print("Data preparation complete.")
    return dataloader

if __name__ == "__main__":
    prepare_data()
