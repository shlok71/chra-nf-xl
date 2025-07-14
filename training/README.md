# LoRA + Distillation-based Training Plan

This document outlines a training plan for a binary student model using LoRA and distillation from a larger teacher model. The training is designed to be feasible on Google Colab Free.

## 1. Teacher Model

We will use a pre-trained GPT-J or Falcon model as the teacher. These models provide a good balance of performance and size, making them suitable for distillation on a free Colab instance.

## 2. Multi-Task Batches

The training data will consist of multi-task batches, including:

- **Text:** General text from sources like Wikipedia and books.
- **OCR:** Text extracted from images using OCR.
- **Canvas Commands:** Instructions for drawing on a 2D canvas (e.g., "draw red circle at 50,50").
- **Web Text:** Text crawled from websites.
- **Reasoning:** Logical and arithmetic problems.

## 3. Quantization

All weights will be quantized to 2-bit ternary or binary using a custom quantization scheme. This will significantly reduce the model size and make it suitable for low-end systems.

## 4. Weight Storage

The quantized weights will be stored as compressed binary files to minimize storage requirements.

## 5. Local Fine-Tuning

The training process will produce a base model that can be fine-tuned locally on specific tasks using LoRA. This allows for efficient customization without requiring a full retraining.
