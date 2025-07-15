import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel, GPT2Config

def main():
    config = GPT2Config.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel(config)
    print("Student model defined.")

if __name__ == "__main__":
    main()
