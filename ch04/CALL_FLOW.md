# Chapter 4: Implementing a GPT Model from Scratch - Call Flow Diagram

This document provides a detailed call flow diagram for the complete GPT model architecture in Chapter 4.

## Overview

Chapter 4 brings together all previous components to build a complete GPT model. It implements:

1. **Layer Normalization** - Stabilizing layer outputs
2. **GELU Activation** - Non-linear activation function
3. **Feed-Forward Network** - Position-wise transformations
4. **Transformer Block** - Complete transformer layer
5. **GPT Model** - Full architecture with embeddings and multiple layers
6. **Text Generation** - Autoregressive generation loop

---

## Complete GPT Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  CH04: IMPLEMENTING GPT MODEL                           │
│                    Complete Architecture Flow                           │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: Token IDs [batch_size, seq_length]
  Example: [8, 1024]
  │
  ▼

┌────────────────────────────────────────────────────────────────────────┐
│  EMBEDDING LAYER                                                       │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► Token Embedding
  │   tok_emb = Embedding(vocab_size, emb_dim)
  │   token_embeddings = tok_emb(token_ids)
  │   │
  │   └─► Shape: [batch, seq_len, emb_dim]
  │       Example: [8, 1024, 768]
  │
  ├─► Positional Embedding
  │   pos_emb = Embedding(context_length, emb_dim)
  │   pos_embeddings = pos_emb(positions)
  │   │
  │   └─► Shape: [batch, seq_len, emb_dim]
  │       Example: [8, 1024, 768]
  │
  ├─► Combine embeddings
  │   x = token_embeddings + pos_embeddings
  │   │
  │   └─► Shape: [8, 1024, 768]
  │
  └─► Apply dropout
      x = emb_dropout(x)


┌────────────────────────────────────────────────────────────────────────┐
│  TRANSFORMER BLOCKS (Repeated N times)                                │
│  For GPT-2 small: N = 12 layers                                       │
└────────────────────────────────────────────────────────────────────────┘

  FOR EACH LAYER (layer = 0 to n_layers-1):
    │
    ├─► Input: x with shape [batch, seq_len, emb_dim]
    │
    ▼

  ┌──────────────────────────────────────────────────────────────────────┐
  │  TRANSFORMER BLOCK (Single Layer)                                    │
  └──────────────────────────────────────────────────────────────────────┘
    │
    ├─► SHORTCUT CONNECTION #1: Save input
    │   shortcut = x
    │
    ├─► Layer Normalization #1
    │   x = LayerNorm(x)
    │   │
    │   └─► Normalize across embedding dimension
    │       Mean = 0, Variance = 1 for each token
    │
    ├─► Multi-Head Attention (from Ch03)
    │   x = MultiHeadAttention(x)
    │   │
    │   │   Inside MultiHeadAttention:
    │   │   ├─► Project to Q, K, V
    │   │   ├─► Split into multiple heads
    │   │   ├─► Compute attention scores
    │   │   ├─► Apply causal mask
    │   │   ├─► Softmax & dropout
    │   │   ├─► Weight values
    │   │   ├─► Concatenate heads
    │   │   └─► Output projection
    │   │
    │   └─► Shape: [batch, seq_len, emb_dim]
    │
    ├─► ADD SHORTCUT #1 (Residual Connection)
    │   x = x + shortcut
    │   │
    │   └─► Allows gradient flow through deep networks
    │
    ├─► SHORTCUT CONNECTION #2: Save input
    │   shortcut = x
    │
    ├─► Layer Normalization #2
    │   x = LayerNorm(x)
    │
    ├─► Feed-Forward Network
    │   x = FeedForward(x)
    │   │
    │   │   Inside FeedForward:
    │   │   ├─► Linear expansion: [emb_dim] → [4*emb_dim]
    │   │   │   x = fc1(x)
    │   │   │   Shape: [batch, seq_len, 4*emb_dim]
    │   │   │
    │   │   ├─► GELU activation (smooth ReLU)
    │   │   │   x = GELU(x)
    │   │   │
    │   │   └─► Linear projection: [4*emb_dim] → [emb_dim]
    │   │       x = fc2(x)
    │   │       Shape: [batch, seq_len, emb_dim]
    │   │
    │   └─► Position-wise transformation (same for each token)
    │
    └─► ADD SHORTCUT #2 (Residual Connection)
        x = x + shortcut

  END TRANSFORMER BLOCK
  │
  └─► Output: x with shape [batch, seq_len, emb_dim]


┌────────────────────────────────────────────────────────────────────────┐
│  OUTPUT HEAD                                                           │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► Final Layer Normalization
  │   x = LayerNorm(x)
  │   │
  │   └─► Shape: [batch, seq_len, emb_dim]
  │       Example: [8, 1024, 768]
  │
  └─► Linear Projection to Vocabulary
      logits = Linear(x, vocab_size)
      │
      └─► Shape: [batch, seq_len, vocab_size]
          Example: [8, 1024, 50257]

OUTPUT: Logits for next token prediction
  Each position has scores for all vocabulary tokens
```

---

## GPT Model Components Hierarchy

```
GPTModel
│
├─── tok_emb: Embedding(vocab_size, emb_dim)
│    └─── Converts token IDs to vectors
│
├─── pos_emb: Embedding(context_length, emb_dim)
│    └─── Adds positional information
│
├─── drop_emb: Dropout(drop_rate)
│    └─── Regularization after embeddings
│
├─── trf_blocks: ModuleList[TransformerBlock × n_layers]
│    │
│    └─── TransformerBlock (repeated 12 times for GPT-2 small)
│         │
│         ├─── norm1: LayerNorm(emb_dim)
│         │    └─── Normalizes before attention
│         │
│         ├─── att: MultiHeadAttention
│         │    ├─── W_query: Linear(emb_dim, emb_dim)
│         │    ├─── W_key: Linear(emb_dim, emb_dim)
│         │    ├─── W_value: Linear(emb_dim, emb_dim)
│         │    ├─── out_proj: Linear(emb_dim, emb_dim)
│         │    ├─── dropout: Dropout(drop_rate)
│         │    └─── mask: Causal mask buffer
│         │
│         ├─── norm2: LayerNorm(emb_dim)
│         │    └─── Normalizes before feed-forward
│         │
│         └─── ff: FeedForward
│              ├─── fc1: Linear(emb_dim, 4*emb_dim)
│              ├─── gelu: GELU()
│              └─── fc2: Linear(4*emb_dim, emb_dim)
│
├─── final_norm: LayerNorm(emb_dim)
│    └─── Final normalization before output
│
└─── out_head: Linear(emb_dim, vocab_size, bias=False)
     └─── Projects to vocabulary for predictions
```

---

## Detailed Component Breakdown

### 1. Layer Normalization

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # x shape: [batch, seq_len, emb_dim]

        # Compute mean and variance across embedding dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift (learnable parameters)
        return self.scale * norm_x + self.shift
```

**Purpose:**
- Stabilizes training by normalizing activations
- Mean = 0, Variance = 1 for each token independently
- Learnable scale and shift parameters

**Flow:**
```
Input: [batch, seq_len, emb_dim]
  ↓
Compute mean per token: [batch, seq_len, 1]
  ↓
Compute variance per token: [batch, seq_len, 1]
  ↓
Normalize: (x - mean) / sqrt(var + eps)
  ↓
Scale and shift: scale * x + shift
  ↓
Output: [batch, seq_len, emb_dim]
```

---

### 2. GELU Activation

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            )
        )
```

**Purpose:**
- Smooth, non-linear activation function
- Similar to ReLU but smoother (differentiable everywhere)
- Used in GPT-2 instead of ReLU

**Comparison:**
```
ReLU:  f(x) = max(0, x)           ← Sharp corner at 0
GELU:  f(x) = x * Φ(x)            ← Smooth everywhere
       where Φ(x) ≈ tanh approximation

For negative values:
  ReLU: Always 0
  GELU: Small negative values allowed

For positive values:
  Both: Similar behavior
```

---

### 3. Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        # x shape: [batch, seq_len, emb_dim]
        return self.layers(x)
```

**Purpose:**
- Position-wise transformation (applied independently to each token)
- Expansion → Non-linearity → Projection
- Adds model capacity beyond attention

**Flow:**
```
Input: [batch, seq_len, 768]
  ↓
Linear expansion: 768 → 3072 (4x)
  [batch, seq_len, 3072]
  ↓
GELU activation
  [batch, seq_len, 3072]
  ↓
Linear projection: 3072 → 768
  [batch, seq_len, 768]
  ↓
Output: [batch, seq_len, 768]
```

**Why 4x expansion?**
- Empirically found to work well
- Creates bottleneck architecture: compress → expand → compress
- Adds expressiveness without changing dimensions

---

### 4. Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # x shape: [batch, seq_len, emb_dim]

        # Multi-head attention block with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        # Feed-forward block with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        return x
```

**Architecture Pattern (Pre-Norm):**
```
Input
  │
  ├──────────────┐ (shortcut)
  │              │
  LayerNorm      │
  │              │
  Attention      │
  │              │
  Dropout        │
  │              │
  └──────(+)─────┘ (add shortcut)
  │
  ├──────────────┐ (shortcut)
  │              │
  LayerNorm      │
  │              │
  FeedForward    │
  │              │
  Dropout        │
  │              │
  └──────(+)─────┘ (add shortcut)
  │
Output
```

**Why Residual Connections?**
- Enable training of very deep networks (100+ layers)
- Gradient flows directly through shortcuts
- Prevents vanishing gradients
- Original input signal preserved

---

### 5. Complete GPT Model

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Embedding layers
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Output layers
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        # in_idx shape: [batch, seq_len]
        batch_size, seq_len = in_idx.shape

        # Token + position embeddings
        tok_embeds = self.tok_emb(in_idx)  # [batch, seq_len, emb_dim]
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )  # [seq_len, emb_dim]
        x = tok_embeds + pos_embeds  # Broadcasting
        x = self.drop_emb(x)

        # Pass through all transformer blocks
        x = self.trf_blocks(x)

        # Final normalization and projection
        x = self.final_norm(x)
        logits = self.out_head(x)  # [batch, seq_len, vocab_size]

        return logits
```

**Configuration Example (GPT-2 Small):**
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Number of tokens in vocabulary
    "context_length": 1024,   # Maximum sequence length
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of transformer blocks
    "drop_rate": 0.1,         # Dropout probability
    "qkv_bias": False         # Use bias in Q/K/V projections
}

# Total parameters: ~124 million
```

---

## Text Generation Flow

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text autoregressively.

    Args:
        model: GPT model
        idx: Initial context [batch, seq_len]
        max_new_tokens: Number of tokens to generate
        context_size: Maximum context length
    """
    # idx is [batch, current_seq_len] array of token indices

    for _ in range(max_new_tokens):
        # Crop context if it exceeds model's context length
        idx_cond = idx[:, -context_size:]

        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)  # [batch, seq_len, vocab_size]

        # Focus only on last time step
        logits = logits[:, -1, :]  # [batch, vocab_size]

        # Get token with highest probability (greedy decoding)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # [batch, 1]

        # Append sampled token to running sequence
        idx = torch.cat([idx, idx_next], dim=1)  # [batch, seq_len+1]

    return idx
```

**Generation Process:**
```
┌────────────────────────────────────────────────────────────────────────┐
│                    AUTOREGRESSIVE TEXT GENERATION                      │
└────────────────────────────────────────────────────────────────────────┘

Initial context: "The cat sat"
Token IDs: [464, 3797, 3332]

ITERATION 1:
  Input:  [464, 3797, 3332]
  │
  ├─► Forward pass through GPT model
  │
  └─► Logits: [..., ..., [0.1, 0.05, ..., 0.8, ...]]  ← 50,257 scores
                              ↑
                     Pick highest: token 319 ("on")

  Updated sequence: [464, 3797, 3332, 319]

ITERATION 2:
  Input:  [464, 3797, 3332, 319]
  │
  ├─► Forward pass through GPT model
  │
  └─► Logits: [..., ..., [..., 0.9, ...]]  ← 50,257 scores
                           ↑
                  Pick highest: token 262 ("the")

  Updated sequence: [464, 3797, 3332, 319, 262]

ITERATION 3:
  Input:  [464, 3797, 3332, 319, 262]
  │
  ├─► Forward pass through GPT model
  │
  └─► Logits: [..., ..., [..., 0.85, ...]]
                           ↑
                  Pick highest: token 2603 ("mat")

  Updated sequence: [464, 3797, 3332, 319, 262, 2603]

...continue until max_new_tokens reached...

Final output: "The cat sat on the mat"
```

---

## Complete Forward Pass Example

```
┌────────────────────────────────────────────────────────────────────────┐
│                   COMPLETE FORWARD PASS EXAMPLE                        │
│                     (GPT-2 Small Configuration)                        │
└────────────────────────────────────────────────────────────────────────┘

INPUT: Token IDs
  Shape: [2, 4]  ← 2 sequences, 4 tokens each
  Values: [[464, 3797, 3332, 319],
           [262, 2603, 373, 1956]]

STEP 1: Embeddings
────────────────────────────────────────────────
  Token embedding:    [2, 4] → [2, 4, 768]
  Position embedding: [4] → [4, 768] → broadcasted to [2, 4, 768]
  Combined:           [2, 4, 768]
  After dropout:      [2, 4, 768]

STEP 2: Transformer Block 1
────────────────────────────────────────────────
  Input:              [2, 4, 768]
  │
  ├─ LayerNorm:       [2, 4, 768]
  ├─ MultiHeadAttn:   [2, 4, 768]
  ├─ + Residual:      [2, 4, 768]
  ├─ LayerNorm:       [2, 4, 768]
  ├─ FeedForward:     [2, 4, 768] → [2, 4, 3072] → [2, 4, 768]
  └─ + Residual:      [2, 4, 768]

STEP 3: Transformer Blocks 2-12
────────────────────────────────────────────────
  (Same structure repeated 11 more times)
  Output:             [2, 4, 768]

STEP 4: Output Head
────────────────────────────────────────────────
  LayerNorm:          [2, 4, 768]
  Linear projection:  [2, 4, 768] → [2, 4, 50257]

OUTPUT: Logits
  Shape: [2, 4, 50257]
  ↑      ↑  ↑  ↑
  │      │  │  └─ Score for each vocabulary token
  │      │  └──── 4 token positions
  └──────┴────── 2 sequences in batch

For next-token prediction:
  - Position 0 predicts token at position 1
  - Position 1 predicts token at position 2
  - Position 2 predicts token at position 3
  - Position 3 predicts token at position 4 (next token)
```

---

## Tensor Shape Reference

| Component | Input Shape | Output Shape | Notes |
|-----------|-------------|--------------|-------|
| **Embeddings** |
| Token Embedding | `[B, T]` | `[B, T, D]` | B=batch, T=seq_len, D=emb_dim |
| Position Embedding | `[T]` | `[T, D]` | Broadcasted to [B, T, D] |
| **Transformer Block** |
| LayerNorm | `[B, T, D]` | `[B, T, D]` | Normalized per token |
| MultiHeadAttention | `[B, T, D]` | `[B, T, D]` | Context aggregation |
| FeedForward | `[B, T, D]` | `[B, T, D]` | Via [B, T, 4D] |
| **Output** |
| Final LayerNorm | `[B, T, D]` | `[B, T, D]` | |
| Output Head | `[B, T, D]` | `[B, T, V]` | V=vocab_size |

**GPT-2 Small Example:**
- B=8, T=1024, D=768, V=50257, H=12 (heads)
- Input: `[8, 1024]`
- Embeddings: `[8, 1024, 768]`
- After 12 blocks: `[8, 1024, 768]`
- Logits: `[8, 1024, 50257]`

---

## Key Design Choices

1. **Pre-Norm Architecture**: LayerNorm before attention/FF (not after)
   - More stable training
   - Better gradient flow

2. **Residual Connections**: Skip connections around each sub-layer
   - Essential for training deep networks
   - Gradient highways

3. **4x FFN Expansion**: FeedForward expands to 4× embedding dimension
   - Adds model capacity
   - Empirically validated

4. **GELU over ReLU**: Smoother activation function
   - Better gradients
   - Slightly better performance

5. **Dropout Regularization**: Applied after attention and FFN
   - Prevents overfitting
   - Typical rate: 0.1

---

## Parameter Count Breakdown (GPT-2 Small: 124M)

```
Component                          Parameters
─────────────────────────────────────────────────────
Token Embedding                    50,257 × 768 = 38.6M
Position Embedding                  1,024 × 768 = 0.8M

Per Transformer Block:
  MultiHeadAttention
    - W_query                         768 × 768 = 0.6M
    - W_key                           768 × 768 = 0.6M
    - W_value                         768 × 768 = 0.6M
    - out_proj                        768 × 768 = 0.6M
  FeedForward
    - fc1 (expand)                  768 × 3072 = 2.4M
    - fc2 (project)                3072 × 768  = 2.4M
  LayerNorm (2x)                     768 × 4 = 3K

  Subtotal per block:                          ~7.2M

12 Transformer Blocks:                        ~86.4M

Output Head                        768 × 50,257 = 38.6M
                                  (shares weights with token embedding)

TOTAL:                                        ~124M
```

---

## Code Location

- **Main notebook**: `ch04/01_main-chapter-code/ch04.ipynb`
- **Standalone script**: `ch04/01_main-chapter-code/gpt.py`
- **Previous chapters**: `ch04/01_main-chapter-code/previous_chapters.py`
- **Exercises**: `ch04/01_main-chapter-code/exercise-solutions.ipynb`

---

## Next Steps

After completing Chapter 4, you'll have:
- ✅ Built complete GPT model architecture
- ✅ Implemented all transformer components
- ✅ Understood residual connections and layer normalization
- ✅ Created basic text generation function

**Ready for Chapter 5**: Training the model from scratch! 🚀
