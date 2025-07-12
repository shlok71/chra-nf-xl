# CHRA-NF-XL: 1B-Parameter CPU-Only Multimodal AI

**CHRA‑NF‑XL** is a highly efficient, 1-billion-parameter multimodal AI model designed to run on modern CPUs (6th-gen Intel Core or newer with AVX2 support) without dedicated GPUs. It aims to deliver sophisticated AI capabilities—from text and image understanding to task automation—within a modest memory footprint of under 12 GB.

## Core Architecture

The model's efficiency and power stem from a novel combination of neural primitives and architectural innovations:

- **High-Dimensional Binary Embeddings (BHV/SHM):** At its core, CHRA-NF-XL uses binary, high-dimensional embeddings and an associative memory system for representing and retrieving information across modalities (text, images, canvas). This is more memory-efficient than traditional float-based embeddings.
- **Sparse Mixture-of-Experts (MoE):** The model employs an **EDGT-XL** transformer architecture, a 16-layer network where only a fraction of experts (4 out of 128) are activated per token. This 80% sparsity, combined with 2-bit weight quantization, drastically reduces computational load.
- **Hybrid Context Management:** A Bloom-sketch-based Key-Value cache provides an effective context window of ~64,000 tokens. For ultra-long contexts, an episodic memory retrieval system fetches relevant information from past interactions.
- **NCA Canvas & DNR:** A Neural Cellular Automata (NCA) canvas enables Optical Character Recognition (OCR) and 2D drawing capabilities, while a Differentiable Neural Renderer (DNR) translates vector graphics into raster images and video.
- **Specialized Experts:** Dedicated experts for **Task Orchestration**, **Tool Invocation**, **Episodic Memory**, and **GUI Management** allow the model to perform complex, asynchronous workflows, interact with external tools (browsers, files), and manage a multi-canvas user interface.

## Performance & Efficiency

CHRA-NF-XL is engineered for high performance on commodity hardware through:
- **On-the-fly 2-bit Quantization:** Weights are quantized to 2 bits during inference, minimizing memory usage.
- **2:4 N:M Sparsity:** Structured sparsity further prunes unnecessary computations.
- **8x8 Block Low-Rank Decomposition:** Reduces the size of weight matrices.

These optimizations collectively enable the model to stay under a 12 GB RAM footprint and achieve inference speeds of less than 2 seconds per 1,000 generated tokens on a target CPU.

## Roadmap

This repository is the monorepo for all C++ and Python components of CHRA-NF-XL. Development will proceed according to the phases outlined in `PHASES.md`.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone --recursive https://github.com/your-org/chra-nf-xl.git
   cd chra-nf-xl
   ```
2. **Run the bootstrap script:**
   - **Linux/macOS:** `cd tools && ./bootstrap.sh`
   - **Windows:** `cd tools && ./bootstrap.ps1`

This will install all necessary dependencies and perform an initial build of the C++ components. See the `docs/` folder for more detailed build and usage instructions.
