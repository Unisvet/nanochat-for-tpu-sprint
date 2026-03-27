# Nanochat: Minimal Decoder-Only Transformer with Flax NNX

This project contains a minimal, decoder-only Transformer implementation (Nanochat) using the official Flax NNX documentation from the TPU Sprint. 

## Overview
Nanochat is designed to be a simple, performant, and educational implementation of a Transformer decoder. It leverages the latest features in Flax NNX, including high-performance JAX transformations and state management.

## Features
- **Flax NNX**: Direct, mutable state management and integration with JAX via the Functional API.
- **Optax Integration**: Efficient training using the standard JAX optimizer library.
- **Minimal Design**: A clean, single-class implementation for researchers and developers to build upon.

## Getting Started

### Prerequisites
- Python 3.10+
- JAX
- Flax (NNX)
- Optax

### Installation
You can initialize the `uv` environment and install dependencies using:
```bash
uv sync
```

### Running the Training Script
Run the following command to start a dummy training run within the `uv` environment:
```bash
uv run train.py
```

## Implementation Details
The model architecture consists of:
- **Token and Positional Embeddings**
- **Stacked Transformer Decoder Blocks** (with Causal Self-Attention and MLP)
- **Layer Normalization**
- **Linear Projection Head**

The training loop uses `nnx.jit` and `nnx.Optimizer` to handle state updates efficiently across JAX transformations.

## License
Apache-2.0
