import os
from datasets import load_dataset, IterableDataset

def download_and_prepare_dataset(dataset_name, split, cache_dir, streaming=True, take_percent=1):
    """Downloads and prepares a single dataset."""
    print(f"Downloading and preparing {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir, streaming=streaming)
        if not isinstance(dataset, IterableDataset) and take_percent < 100:
            dataset = dataset.select(range(int(len(dataset) * (take_percent / 100))))
        # In a real implementation, we would process the dataset here.
        # For now, we will just delete it.
        print(f"Successfully downloaded and prepared {dataset_name}.")
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        print("Skipping this dataset.")

def main():
    """Downloads and prepares all the datasets."""
    datasets_to_download = {
        "mnist": "train",
    }

    cache_dir = "/app/huggingface_cache"

    for dataset_name, split in datasets_to_download.items():
        download_and_prepare_dataset(dataset_name, split, cache_dir, take_percent=10, streaming=False)

if __name__ == "__main__":
    main()
