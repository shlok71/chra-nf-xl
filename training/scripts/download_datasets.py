import os
from datasets import load_dataset

def download_and_prepare_dataset(dataset_name, split, cache_dir):
    """Downloads and prepares a single dataset."""
    print(f"Downloading and preparing {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        # In a real implementation, we would process the dataset here.
        # For now, we will just delete it.
        del dataset
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        print("Skipping this dataset.")

def main():
    """Downloads and prepares all the datasets."""
    datasets_to_download = {
        "tatsu-lab/alpaca": "train",
    }

    cache_dir = "/app/huggingface_cache"

    for dataset_name, split in datasets_to_download.items():
        download_and_prepare_dataset(dataset_name, split, cache_dir)

if __name__ == "__main__":
    main()
