# Nanochat: Minimal Decoder-Only Transformer (TinyShakespeare)

This project contains a minimal, decoder-only Transformer implementation (**Nanochat**) using **Flax NNX** and **Optax**. It is designed to be a high-performance, educational baseline for language modeling tasks.

## Overview
Nanochat is a character-level Transformer that trains on the **TinyShakespeare** dataset. It demonstrates how to use the latest features in Flax NNX, including direct state management and JAX's functional transformations. It is optimized to run locally on a CPU for verification before scaling to cloud accelerators like TPUs.

## Features
- **Flax NNX**: Built with the latest Flax NNX API for modern state management.
- **Character-Level Tokenizer**: Custom character mapping for raw text modeling.
- **TinyShakespeare Dataset**: Automatically downloads and prepares the dataset on first run.
- **CPU Optimization**: Designed to verify math and loss decrease on local hardware.

## Getting Started

### Prerequisites
- Python 3.13+ (or as specified in `pyproject.toml`)
- [uv](https://github.com/astral-sh/uv) (Recommended package manager)

### Installation
Initialize the environment and install all dependencies:
```bash
uv sync
```

### Running the Training
To start training on your local CPU and verify the loss decrease:
```bash
uv run python train.py
```
*Note: The script will download `tinyshakespeare.txt` (approx. 1.1MB) if it's not present.*

## Project Structure
- `model.py`: The core Transformer architecture (Embeddings, Attention, MLP, Decoder Blocks).
- `data_loader.py`: Handles TinyShakespeare download, character mapping, and batching.
- `train.py`: Main training loop with JIT-compiled steps and loss reporting.

## Implementation Details
The model architecture consists of:
- **Token and Positional Embeddings**
- **Stacked Transformer Decoder Blocks** (Causal Self-Attention and MLP)
- **Layer Normalization** and a **Linear Projection Head** for characters.

The training loop calculates cross-entropy loss against character targets and uses **AdamW** for optimization.

## License
Apache-2.0
