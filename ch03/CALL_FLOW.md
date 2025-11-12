# Chapter 3: Coding Attention Mechanisms - Call Flow Diagram

This document provides a detailed call flow diagram for the attention mechanism implementation in Chapter 3.

## Overview

Chapter 3 implements the attention mechanism, which is the core innovation that enables LLMs to process context. This chapter builds on Chapter 2's data pipeline and implements:

1. **Self-Attention** - Computing attention scores and context vectors
2. **Causal Attention** - Adding masking for autoregressive models
3. **Multi-Head Attention** - Parallel attention computations
4. **Scaled Dot-Product Attention** - Normalizing attention scores

---

## Complete Attention Mechanism Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  CH03: CODING ATTENTION MECHANISMS                      │
│                     Attention Computation Flow                          │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: Token Embeddings from Ch02
  Shape: [batch_size, num_tokens, d_in]
  Example: [8, 4, 256]
  │
  │   Each token has a 256-dimensional embedding vector
  │   containing semantic and positional information
  │
  ▼

┌────────────────────────────────────────────────────────────────────────┐
│  STEP 1: LINEAR PROJECTIONS (Query, Key, Value)                       │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► W_query = Linear(d_in, d_out)
  │   queries = W_query(x)
  │   │
  │   └─► Shape: [batch, num_tokens, d_out]
  │       Example: [8, 4, 256]
  │
  ├─► W_key = Linear(d_in, d_out)
  │   keys = W_key(x)
  │   │
  │   └─► Shape: [batch, num_tokens, d_out]
  │       Example: [8, 4, 256]
  │
  └─► W_value = Linear(d_in, d_out)
      values = W_value(x)
      │
      └─► Shape: [batch, num_tokens, d_out]
          Example: [8, 4, 256]

  Q, K, V: Three different "views" of the same input
  - Query: "What am I looking for?"
  - Key:   "What do I contain?"
  - Value: "What information do I have?"


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 2: COMPUTE ATTENTION SCORES (Dot Product)                       │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► attn_scores = queries @ keys.transpose(-2, -1)
  │   │
  │   │   queries: [batch, num_tokens, d_out]
  │   │   keys.T:  [batch, d_out, num_tokens]
  │   │   ──────────────────────────────────────
  │   │   result:  [batch, num_tokens, num_tokens]
  │   │
  │   └─► Shape: [8, 4, 4]
  │       │
  │       │   Each position computes similarity with all positions:
  │       │
  │       │        Token0  Token1  Token2  Token3
  │       │   Token0 [s00    s01    s02    s03]
  │       │   Token1 [s10    s11    s12    s13]
  │       │   Token2 [s20    s21    s22    s23]
  │       │   Token3 [s30    s31    s32    s33]
  │       │
  │       └─► sij = similarity between token i and token j
  │
  └─► Higher scores = more relevant context


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 3: APPLY CAUSAL MASK (for autoregressive models)                │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► Create upper triangular mask:
  │   mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1)
  │   │
  │   │   Example for 4 tokens:
  │   │        0  1  2  3
  │   │   0 [  0  1  1  1 ]  ← Token 0 can't see future tokens
  │   │   1 [  0  0  1  1 ]  ← Token 1 can't see tokens 2,3
  │   │   2 [  0  0  0  1 ]  ← Token 2 can't see token 3
  │   │   3 [  0  0  0  0 ]  ← Token 3 sees all previous
  │   │
  │   └─► 1 = mask (hide), 0 = keep (show)
  │
  ├─► Apply mask to attention scores:
  │   attn_scores.masked_fill_(mask.bool(), -torch.inf)
  │   │
  │   │   Before masking:
  │   │        Token0  Token1  Token2  Token3
  │   │   Token0 [0.5    0.7    0.3    0.9]
  │   │   Token1 [0.4    0.6    0.8    0.2]
  │   │   Token2 [0.3    0.5    0.4    0.7]
  │   │   Token3 [0.6    0.4    0.5    0.8]
  │   │
  │   │   After masking (future tokens = -inf):
  │   │        Token0  Token1  Token2  Token3
  │   │   Token0 [0.5    -inf   -inf   -inf]
  │   │   Token1 [0.4    0.6    -inf   -inf]
  │   │   Token2 [0.3    0.5    0.4    -inf]
  │   │   Token3 [0.6    0.4    0.5    0.8]
  │   │
  │   └─► Prevents token from attending to future positions
  │
  └─► WHY? For next-token prediction, token N shouldn't see N+1!


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 4: SCALE ATTENTION SCORES                                       │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► scaled_scores = attn_scores / sqrt(d_out)
  │   │
  │   └─► WHY? Prevent extremely large values before softmax
  │       - Large dot products → extreme softmax outputs
  │       - Scaling stabilizes training
  │       - sqrt(d_out) is theoretically motivated
  │
  └─► Shape unchanged: [batch, num_tokens, num_tokens]


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 5: SOFTMAX (Convert scores to probabilities)                    │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► attn_weights = softmax(scaled_scores, dim=-1)
  │   │
  │   │   Converts each row to probability distribution:
  │   │
  │   │   After softmax:
  │   │        Token0  Token1  Token2  Token3
  │   │   Token0 [1.0    0.0    0.0    0.0]  ← Only sees itself
  │   │   Token1 [0.3    0.7    0.0    0.0]  ← Mostly focuses on Token1
  │   │   Token2 [0.2    0.3    0.5    0.0]  ← Balanced attention
  │   │   Token3 [0.1    0.2    0.3    0.4]  ← Attends to all
  │   │
  │   │   Properties:
  │   │   - Each row sums to 1.0
  │   │   - All values between 0 and 1
  │   │   - Represents attention distribution
  │   │
  │   └─► Shape: [batch, num_tokens, num_tokens]
  │
  ├─► Optional: Apply dropout for regularization
  │   attn_weights = dropout(attn_weights)
  │
  └─► Attention weights ready for value aggregation


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 6: COMPUTE CONTEXT VECTORS (Weighted sum of values)             │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► context_vec = attn_weights @ values
  │   │
  │   │   attn_weights: [batch, num_tokens, num_tokens]
  │   │   values:       [batch, num_tokens, d_out]
  │   │   ──────────────────────────────────────────────
  │   │   result:       [batch, num_tokens, d_out]
  │   │
  │   └─► Shape: [8, 4, 256]
  │
  ├─► Each position gets weighted combination of all values:
  │   │
  │   │   For token i:
  │   │   context[i] = Σ(attention_weight[i,j] * value[j])
  │   │
  │   │   Example for token 2:
  │   │   context[2] = 0.2*value[0] + 0.3*value[1] + 0.5*value[2]
  │   │
  │   └─► Higher attention weight = more contribution
  │
  └─► OUTPUT: Context-aware representations!

  Each token now contains information from tokens it attended to
```

---

## Multi-Head Attention Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│               MULTI-HEAD ATTENTION (Parallel Processing)               │
└────────────────────────────────────────────────────────────────────────┘

INPUT: [batch, num_tokens, d_in]  (e.g., [8, 4, 256])
  │
  ▼

┌────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Project to Q, K, V                                            │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► W_query(x) → [batch, num_tokens, d_out]
  ├─► W_key(x)   → [batch, num_tokens, d_out]
  └─► W_value(x) → [batch, num_tokens, d_out]


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Reshape for Multiple Heads                                   │
└────────────────────────────────────────────────────────────────────────┘
  │
  │   Given: num_heads = 12, d_out = 768
  │   Then:  head_dim = d_out / num_heads = 64
  │
  ├─► Reshape Q, K, V:
  │   [batch, num_tokens, d_out]
  │   → [batch, num_tokens, num_heads, head_dim]
  │   → [batch, num_heads, num_tokens, head_dim]
  │   │
  │   │   Example: [8, 4, 768]
  │   │   → [8, 4, 12, 64]
  │   │   → [8, 12, 4, 64]
  │   │
  │   └─► Now we have 12 independent attention heads!
  │
  └─► Each head processes a 64-dimensional subspace


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Parallel Attention for Each Head                             │
└────────────────────────────────────────────────────────────────────────┘
  │
  │   FOR EACH HEAD (computed in parallel):
  │
  ├─► Compute attention scores:
  │   attn_scores = Q @ K.transpose(-2, -1)
  │   Shape: [batch, num_heads, num_tokens, num_tokens]
  │   Example: [8, 12, 4, 4]
  │
  ├─► Apply causal mask:
  │   attn_scores.masked_fill_(mask, -inf)
  │
  ├─► Scale and softmax:
  │   attn_weights = softmax(attn_scores / sqrt(head_dim), dim=-1)
  │
  └─► Compute context:
      context = attn_weights @ V
      Shape: [batch, num_heads, num_tokens, head_dim]
      Example: [8, 12, 4, 64]


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Concatenate Heads                                            │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► Transpose back:
  │   [batch, num_heads, num_tokens, head_dim]
  │   → [batch, num_tokens, num_heads, head_dim]
  │   │
  │   Example: [8, 12, 4, 64] → [8, 4, 12, 64]
  │
  ├─► Flatten heads:
  │   [batch, num_tokens, num_heads, head_dim]
  │   → [batch, num_tokens, num_heads * head_dim]
  │   → [batch, num_tokens, d_out]
  │   │
  │   Example: [8, 4, 12, 64] → [8, 4, 768]
  │
  └─► Concatenated output from all heads


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Output Projection                                            │
└────────────────────────────────────────────────────────────────────────┘
  │
  └─► out_proj = Linear(d_out, d_out)
      output = out_proj(concatenated)
      │
      └─► Final Shape: [batch, num_tokens, d_out]
          Example: [8, 4, 768]

OUTPUT: Context-aware representations with multi-head attention!
```

---

## Detailed Code Flow

### Single-Head Causal Self-Attention

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out

        # Linear projections for Q, K, V
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Causal mask (upper triangular)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # x shape: [batch, num_tokens, d_in]
        b, num_tokens, d_in = x.shape

        # STEP 1: Project to Q, K, V
        keys    = self.W_key(x)    # [batch, num_tokens, d_out]
        queries = self.W_query(x)  # [batch, num_tokens, d_out]
        values  = self.W_value(x)  # [batch, num_tokens, d_out]

        # STEP 2: Compute attention scores
        # queries: [b, num_tokens, d_out]
        # keys.T:  [b, d_out, num_tokens]
        # result:  [b, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(1, 2)

        # STEP 3: Apply causal mask
        # Prevent attending to future positions
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )

        # STEP 4: Scale and normalize
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # STEP 5: Compute context vectors
        context_vec = attn_weights @ values
        # [batch, num_tokens, num_tokens] @ [batch, num_tokens, d_out]
        # = [batch, num_tokens, d_out]

        return context_vec
```

### Multi-Head Attention (Efficient Implementation)

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Single linear layers for all heads combined
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # STEP 1: Project to Q, K, V
        keys    = self.W_key(x)    # [b, num_tokens, d_out]
        queries = self.W_query(x)  # [b, num_tokens, d_out]
        values  = self.W_value(x)  # [b, num_tokens, d_out]

        # STEP 2: Reshape for multi-head
        # Split d_out into (num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose to [b, num_heads, num_tokens, head_dim]
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # STEP 3: Compute attention (for all heads in parallel)
        attn_scores = queries @ keys.transpose(2, 3)
        # [b, num_heads, num_tokens, num_tokens]

        # STEP 4: Apply mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # STEP 5: Softmax
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # STEP 6: Compute context vectors
        context_vec = attn_weights @ values
        # [b, num_heads, num_tokens, head_dim]

        # STEP 7: Concatenate heads
        context_vec = context_vec.transpose(1, 2)
        # [b, num_tokens, num_heads, head_dim]

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # [b, num_tokens, d_out]

        # STEP 8: Output projection
        context_vec = self.out_proj(context_vec)

        return context_vec
```

---

## Visual Attention Example

```
┌────────────────────────────────────────────────────────────────────────┐
│              HOW ATTENTION WORKS: CONCRETE EXAMPLE                     │
└────────────────────────────────────────────────────────────────────────┘

Input sentence: "The cat sat on the mat"
Tokens: ["The", "cat", "sat", "on", "the", "mat"]

For token "sat" (position 2):

STEP 1: Compute similarity with all previous tokens
────────────────────────────────────────────────
  Query("sat") · Key("The")  = 0.1
  Query("sat") · Key("cat")  = 0.8
  Query("sat") · Key("sat")  = 0.5
  Query("sat") · Key("on")   = -inf  (masked, future token)
  Query("sat") · Key("the")  = -inf  (masked, future token)
  Query("sat") · Key("mat")  = -inf  (masked, future token)

STEP 2: Apply softmax (convert to probabilities)
────────────────────────────────────────────────
  After softmax:
    "The": 0.1  (10% attention)
    "cat": 0.7  (70% attention)  ← Most attention here!
    "sat": 0.2  (20% attention)

STEP 3: Weighted sum of values
────────────────────────────────────────────────
  context("sat") = 0.1 * Value("The")
                 + 0.7 * Value("cat")
                 + 0.2 * Value("sat")

Result: "sat" now has strong representation of "cat" context!

This makes sense: "cat" is the subject that's doing the sitting.
```

---

## Tensor Shape Transformations

| Stage | Shape | Description |
|-------|-------|-------------|
| Input Embeddings | `[8, 4, 256]` | Batch of 8, 4 tokens, 256-dim |
| After Q/K/V projection | `[8, 4, 256]` | Same shape, different subspace |
| Attention Scores | `[8, 4, 4]` | Token-to-token similarities |
| After Masking | `[8, 4, 4]` | Future positions = -inf |
| Attention Weights | `[8, 4, 4]` | Probability distributions |
| Context Vectors | `[8, 4, 256]` | Weighted combinations |
| **Multi-Head** | | |
| After reshape for heads | `[8, 12, 4, 64]` | 12 heads, 64-dim each |
| Attention per head | `[8, 12, 4, 4]` | Separate attention per head |
| Context per head | `[8, 12, 4, 64]` | Separate context per head |
| After concatenation | `[8, 4, 768]` | All heads combined |
| After output projection | `[8, 4, 768]` | Final multi-head output |

---

## Key Hyperparameters

```python
# Attention dimensions
d_in = 256          # Input dimension (embedding size)
d_out = 256         # Output dimension
num_heads = 12      # Number of attention heads
head_dim = 64       # d_out / num_heads

# Context settings
context_length = 1024   # Maximum sequence length
dropout = 0.1           # Dropout rate for regularization

# Optional settings
qkv_bias = False    # Whether to use bias in Q/K/V projections
```

---

## Why Multi-Head Attention?

**Single Head:**
- Learns one type of relationship
- Limited representational capacity

**Multi-Head:**
- Each head can specialize in different patterns:
  - Head 1: Subject-verb relationships
  - Head 2: Noun-adjective relationships
  - Head 3: Long-range dependencies
  - Head 4: Local context
  - etc.
- More expressive
- Better generalization

---

## Important Implementation Details

1. **Causal Masking**: Essential for autoregressive models (GPT). Prevents information leakage from future tokens.

2. **Scaling Factor**: `1/sqrt(d_out)` prevents softmax saturation with large embedding dimensions.

3. **Dropout**: Applied to attention weights for regularization, not to context vectors directly.

4. **Buffer vs Parameter**: Mask is registered as buffer (not updated during training), Q/K/V are parameters.

5. **Efficient Multi-Head**: Single projection matrix split into heads is more efficient than multiple separate projections.

---

## Complete Pipeline

```
Input Text
    ↓
[Ch02: Tokenization & Embedding]
    ↓
Token Embeddings [batch, tokens, d_in]
    ↓
[Ch03: Multi-Head Attention]
    ├─► Linear Projection to Q, K, V
    ├─► Reshape for Multiple Heads
    ├─► Compute Attention Scores
    ├─► Apply Causal Mask
    ├─► Scale and Softmax
    ├─► Weighted Sum of Values
    ├─► Concatenate Heads
    └─► Output Projection
    ↓
Context Vectors [batch, tokens, d_out]
    ↓
Ready for Transformer Block (Ch04)! 🚀
```

---

## Code Location

- **Main notebook**: `ch03/01_main-chapter-code/ch03.ipynb`
- **Condensed version**: `ch03/01_main-chapter-code/multihead-attention.ipynb`
- **Exercises**: `ch03/01_main-chapter-code/exercise-solutions.ipynb`

---

## Next Steps

After completing Chapter 3, you'll have:
- ✅ Implemented self-attention mechanism
- ✅ Added causal masking for autoregressive modeling
- ✅ Built multi-head attention for richer representations
- ✅ Understood scaled dot-product attention

**Ready for Chapter 4**: Building the complete GPT model! 🚀
