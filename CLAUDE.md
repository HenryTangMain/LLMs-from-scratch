# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains the code for "Build a Large Language Model (From Scratch)" by Sebastian Raschka. It implements a GPT-like LLM from scratch in PyTorch, covering the full pipeline from data preprocessing and tokenization through pretraining and finetuning.

**Important**: This is a companion repository to a published book. Do not extend or modify the main chapter code in ways that would deviate from the book's content. The code is designed to be educational and runs on conventional laptops without specialized hardware.

## Repository Structure

The repository is organized by book chapters:

- **ch02-ch07**: Main chapter code implementing progressive LLM concepts
  - Each chapter has `01_main-chapter-code/` containing the primary implementations
  - Additional subdirectories contain bonus/optional material
- **appendix-A through appendix-E**: Supplementary materials
- **pkg/llms_from_scratch**: Installable PyPI package for convenient imports
- **setup/**: Environment setup guides and Docker configurations
- **reasoning-from-scratch/**: Companion repository code for the sequel book

### Key Code Architecture

**Progressive Chapter Structure**: Each chapter builds on previous work using `previous_chapters.py` files that consolidate relevant code from earlier chapters. This avoids repetition while maintaining self-contained examples.

**Main Model Implementations**:
- `ch04/01_main-chapter-code/gpt.py`: Core GPT model architecture (GPTModel, MultiHeadAttention, TransformerBlock)
- `ch05/01_main-chapter-code/gpt_train.py`: Training loop and utilities
- `ch06/01_main-chapter-code/gpt_class_finetune.py`: Classification finetuning
- `ch07/01_main-chapter-code/gpt_instruction_finetuning.py`: Instruction finetuning

**Data Pipeline** (ch02):
- Tokenization using tiktoken (BPE encoding)
- `GPTDatasetV1`: Sliding window dataset for autoregressive training
- `create_dataloader_v1()`: Standard PyTorch DataLoader creation

**PyPI Package** (`pkg/llms_from_scratch`):
Modular chapter-based imports. Each chapter's code is consolidated into importable modules (ch02.py, ch03.py, ch04.py, etc.). Also includes bonus implementations:
- `llama3.py`: Llama 3 implementation
- `qwen3.py`: Qwen3 implementation
- `kv_cache/`: KV cache optimizations
- `kv_cache_batched/`: Batched inference with KV cache

## Development Commands

### Environment Setup

**Quick start** (from repository root):
```bash
pip install -r requirements.txt
```

**Using uv** (recommended package manager):
```bash
# Install dependencies
uv sync

# Install with bonus materials
uv sync --group bonus

# Install with dev dependencies
uv sync --dev
```

**Install as editable package**:
```bash
pip install -e .
# or with uv:
uv add --editable . --dev
```

### Running Tests

Tests are located in various chapter directories and use pytest:

```bash
# Activate virtual environment first (if using uv sync)
source .venv/bin/activate

# Run specific chapter tests
pytest ch04/01_main-chapter-code/tests.py
pytest ch05/01_main-chapter-code/tests.py
pytest ch06/01_main-chapter-code/tests.py

# Run bonus material tests
pytest ch05/07_gpt_to_llama/tests/test_llama32_nb.py
pytest ch05/11_qwen3/tests/test_qwen3_nb.py
pytest ch05/12_gemma3/tests/test_gemma3_nb.py

# Run package tests
pytest pkg/llms_from_scratch/tests/

# Validate notebooks
pytest --nbval ch02/01_main-chapter-code/dataloader.ipynb
pytest --nbval ch03/01_main-chapter-code/multihead-attention.ipynb
```

### Running Training Scripts

Training scripts are standalone Python files:

```bash
# Pretrain GPT model
cd ch05/01_main-chapter-code
python gpt_train.py

# Generate text from pretrained model
python gpt_generate.py

# Finetune for classification
cd ch06/01_main-chapter-code
python gpt_class_finetune.py

# Instruction finetuning
cd ch07/01_main-chapter-code
python gpt_instruction_finetuning.py
```

### Working with Jupyter Notebooks

All main chapters include `.ipynb` files demonstrating the concepts:

```bash
jupyter lab
# Then open ch0X/01_main-chapter-code/ch0X.ipynb
```

### Linting

Configured with ruff (see `pyproject.toml`):

```bash
# Configuration in pyproject.toml
# Line length: 140
# Some errors ignored for educational code readability
```

## Key Technical Details

### Model Configuration

GPT model is configured via a dictionary (see ch04, ch05):
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

### Device Handling

Code automatically uses GPU if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Weight Loading

The repository supports loading pretrained weights from:
- OpenAI GPT-2 (ch05)
- Meta Llama 3/3.1/3.2 (ch05/07_gpt_to_llama)
- Qwen3 (ch05/11_qwen3)
- Gemma 3 (ch05/12_gemma3)

See `load_weights_into_gpt()` and related functions in ch05 for weight conversion utilities.

### Common Utilities

**Text generation**:
- `generate_text_simple()`: Basic greedy decoding (ch04)
- `generate()`: Advanced generation with temperature, top-k sampling (ch05)

**Training utilities**:
- `calc_loss_batch()`: Compute cross-entropy loss for a batch
- `calc_loss_loader()`: Average loss over entire dataloader
- `evaluate_model()`: Compute train/val losses
- `train_model_simple()`: Main training loop

**Tokenization**:
- `text_to_token_ids()`: Encode text to token IDs
- `token_ids_to_text()`: Decode token IDs to text

## Python Version & Dependencies

- **Python**: 3.10 to 3.13
- **PyTorch**: ≥2.2.2 (version varies by platform, see pyproject.toml)
- **Key dependencies**: tiktoken, matplotlib, tensorflow (for data loading), tqdm, pandas

Platform-specific torch/tensorflow versions are handled in `pyproject.toml` with conditional dependencies.

## Important Conventions

1. **previous_chapters.py**: Each chapter includes this file consolidating code from earlier chapters. Import from here to get previously-implemented functionality.

2. **Notebook vs. Script**: Main concepts are in notebooks (ch0X.ipynb), with summarized standalone scripts (e.g., gpt.py, gpt_train.py) for easy reuse.

3. **Tests**: Simple sanity tests in `tests.py` files verify that models can forward pass and produce expected tensor shapes.

4. **Data Loading**: Training data is either downloaded automatically (e.g., Project Gutenberg) or loaded from included JSON files (e.g., instruction-data.json).

5. **File Paths**: Many scripts assume execution from their containing directory. Use absolute paths or cd into the chapter directory first.
