# Chapter 2: Working with Text Data - Call Flow Diagram

This document provides a detailed call flow diagram for the data preprocessing pipeline in Chapter 2.

## Overview

Chapter 2 implements the complete data pipeline that transforms raw text into embedded tensors ready for LLM training. The pipeline consists of:

1. **Tokenization** - Converting text to token IDs
2. **Dataset Creation** - Creating input/target pairs with sliding windows
3. **Batching** - Grouping samples into batches
4. **Embedding** - Converting token IDs to continuous vectors
5. **Position Encoding** - Adding positional information

---

## Complete Data Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CH02: WORKING WITH TEXT DATA                         │
│                        Data Pipeline Flow                               │
└─────────────────────────────────────────────────────────────────────────┘

1. RAW TEXT INPUT
   │
   ├─► "the-verdict.txt" (or any text file)
   │   └─► Read file content as string
   │
   └─► raw_text = "I HAD always thought Jack Gisburn..."


2. TOKENIZATION (Text → Token IDs)
   │
   ├─► tiktoken.get_encoding("gpt2")
   │   └─► BPE Tokenizer initialized
   │
   ├─► tokenizer.encode(text, allowed_special={"<|endoftext|>"})
   │   │
   │   ├─► Breaks text into subwords/characters
   │   ├─► Maps each token to integer ID
   │   └─► Returns: [40, 367, 2885, 1464, 1807, ...]
   │
   └─► token_ids (List of integers)


3. DATASET CREATION (Sliding Window)
   │
   ├─► GPTDatasetV1(txt, tokenizer, max_length, stride)
   │   │
   │   ├─► __init__(txt, tokenizer, max_length, stride):
   │   │   │
   │   │   ├─► Encode entire text to token_ids
   │   │   │
   │   │   ├─► Sliding window loop:
   │   │   │   for i in range(0, len(token_ids)-max_length, stride):
   │   │   │       │
   │   │   │       ├─► input_chunk = token_ids[i : i+max_length]
   │   │   │       │   └─► [290, 4920, 2241, 287]
   │   │   │       │
   │   │   │       └─► target_chunk = token_ids[i+1 : i+max_length+1]
   │   │   │           └─► [4920, 2241, 287, 257]  (shifted by 1)
   │   │   │
   │   │   └─► Store as tensors in self.input_ids & self.target_ids
   │   │
   │   ├─► __len__(): Returns number of samples
   │   │
   │   └─► __getitem__(idx): Returns (input_ids[idx], target_ids[idx])
   │
   └─► Dataset object ready


4. DATALOADER CREATION (Batching)
   │
   ├─► create_dataloader_v1(txt, batch_size, max_length, stride, ...)
   │   │
   │   ├─► Initialize tokenizer
   │   │
   │   ├─► Create GPTDatasetV1 (from step 3)
   │   │
   │   └─► DataLoader(dataset, batch_size, shuffle, drop_last, ...)
   │       │
   │       └─► Returns batched samples
   │           ├─► inputs:  [batch_size, max_length]
   │           └─► targets: [batch_size, max_length]
   │
   └─► DataLoader object (iterable)


5. ITERATION & BATCHING
   │
   ├─► for batch in dataloader:
   │       x, y = batch
   │
   │   Example with batch_size=8, max_length=4:
   │       x shape: torch.Size([8, 4])  ← 8 samples, 4 tokens each
   │       y shape: torch.Size([8, 4])  ← targets (shifted by 1)
   │
   └─► Batched token IDs ready for embedding


6. TOKEN EMBEDDING (IDs → Vectors)
   │
   ├─► token_embedding_layer = nn.Embedding(vocab_size, output_dim)
   │   │                                    (50257,      256)
   │   │
   │   └─► Embedding matrix: [50257 × 256]
   │       Each token ID maps to a 256-dim vector
   │
   ├─► token_embeddings = token_embedding_layer(x)
   │   │
   │   └─► Input:  [8, 4]      (batch_size, max_length)
   │       Output: [8, 4, 256] (batch_size, max_length, embedding_dim)
   │
   └─► Token vectors created


7. POSITIONAL EMBEDDING (Position → Vectors)
   │
   ├─► pos_embedding_layer = nn.Embedding(context_length, output_dim)
   │   │                                  (1024,          256)
   │   │
   │   └─► Position embedding matrix: [1024 × 256]
   │       Each position (0-1023) has a 256-dim vector
   │
   ├─► pos_embeddings = pos_embedding_layer(torch.arange(max_length))
   │   │                                    [0, 1, 2, 3]
   │   │
   │   └─► Input:  [4]         (max_length)
   │       Output: [4, 256]    (max_length, embedding_dim)
   │
   └─► Position vectors created


8. FINAL INPUT EMBEDDINGS (Combine Token + Position)
   │
   ├─► input_embeddings = token_embeddings + pos_embeddings
   │   │
   │   │   token_embeddings: [8, 4, 256]
   │   │   pos_embeddings:   [4, 256]     (broadcasted to [8, 4, 256])
   │   │   ────────────────────────────
   │   │   result:           [8, 4, 256]
   │   │
   │   └─► Each token now has:
   │       ├─► Semantic information (from token embedding)
   │       └─► Position information (from positional embedding)
   │
   └─► READY FOR LLM INPUT! ✓
```

---

## Key Components

### GPTDatasetV1 Class

A PyTorch Dataset that implements the sliding window approach for creating training samples.

**Methods:**
- `__init__(txt, tokenizer, max_length, stride)`: Create input/target pairs with sliding window
- `__len__()`: Return number of samples
- `__getitem__(idx)`: Get sample by index

**Sliding Window Logic:**
```
Input sequence:  [A, B, C, D, E, F, G, H, I, J]
max_length = 4, stride = 4

Window 1:
  Input:  [A, B, C, D]
  Target: [B, C, D, E]

Window 2 (stride=4, starts at position 4):
  Input:  [E, F, G, H]
  Target: [F, G, H, I]
```

### create_dataloader_v1() Function

Convenience function that encapsulates the entire preprocessing pipeline.

**Parameters:**
- `txt`: Raw text string
- `batch_size`: Number of samples per batch
- `max_length`: Sequence length (context window)
- `stride`: Step size for sliding window
- `shuffle`: Whether to shuffle data
- `drop_last`: Drop incomplete final batch
- `num_workers`: Number of parallel workers

**Returns:** PyTorch DataLoader object

---

## Detailed Function Call Sequence

```python
# ============================================
# STEP 1: Load Raw Text
# ============================================
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# Result: String containing full text


# ============================================
# STEP 2: Create DataLoader (encapsulates preprocessing)
# ============================================
dataloader = create_dataloader_v1(
    txt=raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=True
)

# Internal execution flow:
#
# 2a. Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
#
# 2b. Create dataset with sliding window
dataset = GPTDatasetV1(raw_text, tokenizer, max_length=4, stride=4)
#     │
#     ├─► token_ids = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
#     │   # Example: [40, 367, 2885, 1464, 1807, 3619, 402, 271, ...]
#     │
#     └─► for i in range(0, len(token_ids) - max_length, stride):
#             input_chunk = token_ids[i : i + max_length]
#             target_chunk = token_ids[i + 1 : i + max_length + 1]
#             self.input_ids.append(torch.tensor(input_chunk))
#             self.target_ids.append(torch.tensor(target_chunk))
#
# 2c. Wrap in DataLoader for batching
return DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    drop_last=True,
    num_workers=0
)


# ============================================
# STEP 3: Create Embedding Layers
# ============================================
vocab_size = 50257      # GPT-2 vocabulary size
output_dim = 256        # Embedding dimension
context_length = 1024   # Maximum sequence length

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)


# ============================================
# STEP 4: Iterate and Embed
# ============================================
for batch in dataloader:
    x, y = batch
    # x: torch.Size([8, 4])  - 8 samples, 4 tokens each
    # y: torch.Size([8, 4])  - targets (shifted by 1 position)

    # 4a. Convert token IDs to vectors
    token_embeddings = token_embedding_layer(x)
    # Input:  [8, 4]      (batch_size, max_length)
    # Output: [8, 4, 256] (batch_size, max_length, embedding_dim)

    # 4b. Get positional embeddings
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    # Input:  [4]      (sequence positions: 0, 1, 2, 3)
    # Output: [4, 256] (max_length, embedding_dim)

    # 4c. Combine token + position embeddings
    input_embeddings = token_embeddings + pos_embeddings
    # token_embeddings: [8, 4, 256]
    # pos_embeddings:   [4, 256]     <- broadcasted to [8, 4, 256]
    # result:           [8, 4, 256]

    # ✓ Ready for Transformer model!
    break  # Process first batch only for this example
```

---

## Visual Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA TRANSFORMATION                            │
└─────────────────────────────────────────────────────────────────────┘

Raw Text String
"I HAD always thought Jack Gisburn rather a cheap genius..."
      ↓
[Tokenization: tiktoken BPE]
      ↓
Token IDs (integers)
[40, 367, 2885, 1464, 1807, 3619, 402, 271, 10899, ...]
      ↓
[Sliding Window: GPTDatasetV1]
Creates overlapping input/target pairs
      ↓
Dataset Samples
Sample 0: input=[40, 367, 2885, 1464], target=[367, 2885, 1464, 1807]
Sample 1: input=[1807, 3619, 402, 271], target=[3619, 402, 271, 10899]
Sample 2: input=[10899, 2138, 257, 7026], target=[2138, 257, 7026, 15632]
...
      ↓
[Batching: PyTorch DataLoader]
Groups samples into batches
      ↓
Batched Tensors
x: torch.Size([8, 4])  - batch of 8 sequences, 4 tokens each
y: torch.Size([8, 4])  - corresponding targets
      ↓
[Token Embedding Layer]
Maps each token ID to a learned vector
      ↓
Token Embeddings
torch.Size([8, 4, 256])  - each token → 256-dim vector
      ↓
[+ Positional Embedding Layer]
Adds position-specific information
      ↓
Final Input Embeddings
torch.Size([8, 4, 256])  - token info + position info
      ↓
Ready for Transformer! 🚀
```

---

## Token ID to Embedding Lookup Example

```
┌────────────────────────────────────────────────────────────────────┐
│            HOW TOKEN EMBEDDING WORKS                               │
└────────────────────────────────────────────────────────────────────┘

Given:
  vocab_size = 50257
  output_dim = 256

Embedding Layer:
  token_embedding_layer = nn.Embedding(50257, 256)

  Creates a matrix of shape [50257, 256]:

  Token ID │ Embedding Vector (256 dimensions)
  ─────────┼────────────────────────────────────────
     0     │ [0.123, -0.456, 0.789, ..., 0.234]
     1     │ [-0.234, 0.567, -0.890, ..., 0.345]
     2     │ [0.345, -0.678, 0.901, ..., -0.456]
    ...    │ ...
    40     │ [0.456, 0.789, -0.123, ..., 0.567]
    ...    │ ...
   50256   │ [-0.567, 0.890, 0.234, ..., -0.678]

Lookup Process:
  Input token IDs: [40, 367, 2885, 1464]

  Each ID fetches its corresponding row:
  ID 40    → embedding_layer.weight[40]    = [0.456, 0.789, ...]
  ID 367   → embedding_layer.weight[367]   = [0.234, -0.123, ...]
  ID 2885  → embedding_layer.weight[2885]  = [-0.890, 0.456, ...]
  ID 1464  → embedding_layer.weight[1464]  = [0.678, -0.234, ...]

  Result: tensor of shape [4, 256]
```

---

## Positional Embedding Example

```
┌────────────────────────────────────────────────────────────────────┐
│          WHY WE NEED POSITIONAL EMBEDDINGS                         │
└────────────────────────────────────────────────────────────────────┘

Problem:
  Without position info, "cat ate mouse" and "mouse ate cat"
  would have identical embeddings (just different order).

Solution:
  Add position-specific vectors to each token embedding.

Example:
  Sentence: "The cat sat"
  Tokens:   ["The", "cat", "sat"]
  Token IDs: [464, 3797, 3332]

  Position embeddings for positions [0, 1, 2]:

  Position 0 → [0.123, -0.456, 0.789, ..., 0.234]
  Position 1 → [-0.234, 0.567, -0.890, ..., 0.345]
  Position 2 → [0.345, -0.678, 0.901, ..., -0.456]

  Final embeddings = token_embeddings + pos_embeddings

  "The" at position 0:
    token_emb[464] + pos_emb[0] = combined vector

  "cat" at position 1:
    token_emb[3797] + pos_emb[1] = combined vector

  "sat" at position 2:
    token_emb[3332] + pos_emb[2] = combined vector
```

---

## Tensor Shape Transformations Summary

| Stage | Input Shape | Output Shape | Description |
|-------|-------------|--------------|-------------|
| Raw Text | String | - | "I HAD always thought..." |
| Tokenization | String | `[seq_len]` | `[40, 367, 2885, ...]` |
| Dataset Creation | `[seq_len]` | Multiple `[max_length]` pairs | Sliding window chunks |
| DataLoader Batching | `[max_length]` | `[batch_size, max_length]` | `[8, 4]` |
| Token Embedding | `[8, 4]` | `[8, 4, 256]` | Each ID → 256-dim vector |
| Positional Embedding | `[4]` | `[4, 256]` | Position → 256-dim vector |
| Final Embedding | `[8, 4, 256]` + `[4, 256]` | `[8, 4, 256]` | Broadcasting addition |

---

## Key Hyperparameters

```python
# Text preprocessing
max_length = 4        # Context window size (tokens per sample)
stride = 4            # Sliding window step size
batch_size = 8        # Samples per batch

# Embedding dimensions
vocab_size = 50257    # GPT-2 BPE vocabulary size
output_dim = 256      # Embedding vector dimension
context_length = 1024 # Maximum sequence length

# DataLoader settings
shuffle = True        # Randomize sample order
drop_last = True      # Drop incomplete final batch
num_workers = 0       # Number of parallel data loading workers
```

---

## Important Notes

1. **Targets are shifted inputs**: Target sequence is input shifted by 1 position to the right. This enables next-token prediction training.

2. **Sliding window overlap**: When `stride < max_length`, windows overlap, providing more training samples but potentially causing overfitting.

3. **Broadcasting in embedding addition**: When adding `[8, 4, 256]` + `[4, 256]`, PyTorch automatically broadcasts the second tensor across the batch dimension.

4. **Byte Pair Encoding (BPE)**: GPT-2 uses BPE tokenization which breaks unknown words into subword units, eliminating the need for `<UNK>` tokens.

5. **Special tokens**: GPT-2 uses `<|endoftext|>` to mark boundaries between different text sources and for padding.

---

## Code Location

- **Main notebook**: `ch02/01_main-chapter-code/ch02.ipynb`
- **Condensed version**: `ch02/01_main-chapter-code/dataloader.ipynb`
- **Exercises**: `ch02/01_main-chapter-code/exercise-solutions.ipynb`

---

## Next Steps

After completing Chapter 2, you'll have:
- ✅ Tokenized text using BPE
- ✅ Created training datasets with sliding windows
- ✅ Built PyTorch DataLoaders for batching
- ✅ Embedded tokens in continuous vector space
- ✅ Added positional information to embeddings

**Ready for Chapter 3**: Implementing attention mechanisms! 🚀
