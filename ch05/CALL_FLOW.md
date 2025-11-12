# Chapter 5: Pretraining on Unlabeled Data - Call Flow Diagram

This document provides a detailed call flow diagram for the training pipeline in Chapter 5.

## Overview

Chapter 5 implements the complete training loop for pretraining a GPT model on unlabeled text data. It covers:

1. **Loss Calculation** - Computing cross-entropy loss for next-token prediction
2. **Training Loop** - Iterative optimization process
3. **Evaluation** - Monitoring training and validation performance
4. **Weight Loading** - Loading pretrained GPT-2 weights
5. **Advanced Generation** - Temperature sampling and top-k sampling
6. **Model Saving/Loading** - Checkpointing trained models

---

## Complete Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  CH05: PRETRAINING ON UNLABELED DATA                    │
│                        Training Pipeline Flow                           │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  STEP 1: SETUP                                                         │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► Load and prepare data
  │   with open("the-verdict.txt", "r") as f:
  │       text_data = f.read()
  │
  ├─► Split into train/validation
  │   train_ratio = 0.90
  │   split_idx = int(train_ratio * len(text_data))
  │   train_data = text_data[:split_idx]
  │   val_data = text_data[split_idx:]
  │
  ├─► Create dataloaders (from Ch02)
  │   train_loader = create_dataloader_v1(
  │       train_data, batch_size=2, max_length=256, stride=256
  │   )
  │   val_loader = create_dataloader_v1(
  │       val_data, batch_size=2, max_length=256, stride=256
  │   )
  │
  ├─► Initialize model (from Ch04)
  │   model = GPTModel(GPT_CONFIG_124M)
  │   model.to(device)
  │
  └─► Initialize optimizer
      optimizer = torch.optim.AdamW(
          model.parameters(),
          lr=0.0004,
          weight_decay=0.1
      )


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 2: TRAINING LOOP (Main)                                         │
└────────────────────────────────────────────────────────────────────────┘

  FOR epoch IN range(num_epochs):
    │
    ├─► Set model to training mode
    │   model.train()
    │
    │   FOR batch IN train_loader:
    │     │
    │     ├─► Get input and target from batch
    │     │   input_batch, target_batch = batch
    │     │   │
    │     │   │   input_batch:  [batch_size, seq_len]
    │     │   │   target_batch: [batch_size, seq_len]
    │     │   │   (targets = inputs shifted by 1)
    │     │
    │     ├─► FORWARD PASS
    │     │   │
    │     │   ├─► Zero gradients from previous step
    │     │   │   optimizer.zero_grad()
    │     │   │
    │     │   ├─► Forward through model
    │     │   │   logits = model(input_batch)
    │     │   │   │
    │     │   │   └─► Shape: [batch_size, seq_len, vocab_size]
    │     │   │
    │     │   └─► Compute loss
    │     │       loss = calc_loss_batch(input_batch, target_batch, model)
    │     │
    │     ├─► BACKWARD PASS
    │     │   │
    │     │   ├─► Compute gradients
    │     │   │   loss.backward()
    │     │   │
    │     │   └─► Update weights
    │     │       optimizer.step()
    │     │
    │     └─► Optional: Evaluate every N steps
    │         if global_step % eval_freq == 0:
    │             evaluate_model(model, train_loader, val_loader)
    │
    └─► Optional: Generate sample text after epoch
        generate_and_print_sample(model, tokenizer, device, "Every effort moves you")


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 3: LOSS CALCULATION (Next-Token Prediction)                     │
└────────────────────────────────────────────────────────────────────────┘

def calc_loss_batch(input_batch, target_batch, model, device):
    │
    ├─► Move data to device
    │   input_batch = input_batch.to(device)
    │   target_batch = target_batch.to(device)
    │
    ├─► Forward pass
    │   logits = model(input_batch)
    │   │
    │   └─► Shape: [batch_size, seq_len, vocab_size]
    │       Example: [8, 256, 50257]
    │
    ├─► Flatten for loss computation
    │   logits_flat = logits.flatten(0, 1)
    │   targets_flat = target_batch.flatten()
    │   │
    │   │   Before: logits  [8, 256, 50257]
    │   │           targets [8, 256]
    │   │
    │   │   After:  logits  [2048, 50257]  ← 8*256 = 2048
    │   │           targets [2048]
    │   │
    │   └─► Treat all positions as independent predictions
    │
    └─► Compute cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        │
        └─► Measures how well model predicts next token

    return loss


┌────────────────────────────────────────────────────────────────────────┐
│  CROSS-ENTROPY LOSS DETAILS                                           │
└────────────────────────────────────────────────────────────────────────┘

For each token position:
  │
  ├─► Model outputs logits (unnormalized scores)
  │   logits = [s₀, s₁, s₂, ..., s₅₀₂₅₆]  ← score for each vocab token
  │
  ├─► Convert to probabilities
  │   probs = softmax(logits)
  │   probs = [p₀, p₁, p₂, ..., p₅₀₂₅₆]  ← sums to 1
  │
  ├─► Compare to actual next token
  │   target = 3797  (e.g., token ID for "cat")
  │
  └─► Compute loss
      loss = -log(probs[target])
      │
      │   If probs[3797] = 0.8  → loss = -log(0.8) = 0.22  ← good
      │   If probs[3797] = 0.1  → loss = -log(0.1) = 2.30  ← bad
      │   If probs[3797] = 0.01 → loss = -log(0.01) = 4.61 ← very bad
      │
      └─► Lower loss = better prediction
```

---

## Evaluation Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│  MODEL EVALUATION                                                      │
└────────────────────────────────────────────────────────────────────────┘

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    │
    ├─► Set model to evaluation mode
    │   model.eval()
    │   │
    │   └─► Disables dropout, puts BatchNorm in eval mode
    │
    ├─► Disable gradient computation (saves memory)
    │   with torch.no_grad():
    │       │
    │       ├─► Evaluate on training data
    │       │   train_loss = calc_loss_loader(
    │       │       train_loader, model, device, num_batches=eval_iter
    │       │   )
    │       │
    │       └─► Evaluate on validation data
    │           val_loss = calc_loss_loader(
    │               val_loader, model, device, num_batches=eval_iter
    │           )
    │
    ├─► Set model back to training mode
    │   model.train()
    │
    └─► Return losses
        return train_loss, val_loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Compute average loss over entire dataloader"""
    │
    ├─► Initialize accumulator
    │   total_loss = 0.
    │
    ├─► Iterate through batches
    │   for i, (input_batch, target_batch) in enumerate(data_loader):
    │       │
    │       ├─► Stop after num_batches (if specified)
    │       │   if i >= num_batches:
    │       │       break
    │       │
    │       ├─► Compute loss for this batch
    │       │   loss = calc_loss_batch(input_batch, target_batch, model, device)
    │       │
    │       └─► Accumulate
    │           total_loss += loss.item()
    │
    └─► Return average
        return total_loss / num_batches
```

---

## Advanced Text Generation

### Temperature Sampling

```python
def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    """
    Generate text with temperature and top-k sampling.

    Temperature controls randomness:
      - temperature < 1: More conservative (peaked distribution)
      - temperature = 1: Use raw probabilities
      - temperature > 1: More random (flatter distribution)

    Top-k limits sampling to k most likely tokens
    """
    for _ in range(max_new_tokens):
        # Crop context to model's max length
        idx_cond = idx[:, -context_size:]

        # Get logits
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # Focus on last position

        # Apply top-k filtering
        if top_k is not None:
            # Keep only top k logits, set others to -inf
            top_logits, top_indices = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]  # k-th highest value
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Convert to probabilities and sample
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding (temperature = 0)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append to sequence
        idx = torch.cat([idx, idx_next], dim=1)

    return idx
```

**Temperature Effect:**
```
Original logits: [2.0, 1.0, 0.5, 0.1]

Temperature = 0.5 (more peaked):
  Scaled: [4.0, 2.0, 1.0, 0.2]
  Probs:  [0.73, 0.20, 0.06, 0.01]  ← More confident

Temperature = 1.0 (unchanged):
  Scaled: [2.0, 1.0, 0.5, 0.1]
  Probs:  [0.52, 0.19, 0.11, 0.07]  ← Balanced

Temperature = 2.0 (flatter):
  Scaled: [1.0, 0.5, 0.25, 0.05]
  Probs:  [0.38, 0.23, 0.18, 0.15]  ← More random
```

**Top-k Sampling:**
```
All logits: [2.0, 1.5, 1.0, 0.8, 0.5, 0.3, ...]
             ↑    ↑    ↑    ↑
             Keep top-4, set rest to -inf

After top-k (k=4):
  Filtered: [2.0, 1.5, 1.0, 0.8, -inf, -inf, ...]
  Probs:    [0.42, 0.28, 0.17, 0.13, 0.0, 0.0, ...]

Sample only from top 4 tokens
```

---

## Loading Pretrained Weights

```
┌────────────────────────────────────────────────────────────────────────┐
│  LOADING PRETRAINED GPT-2 WEIGHTS                                     │
└────────────────────────────────────────────────────────────────────────┘

def load_weights_into_gpt(gpt, params):
    """
    Load weights from pretrained GPT-2 checkpoint.

    params: Dictionary with structure:
      {
        "wte": token_embedding_weights,
        "wpe": position_embedding_weights,
        "blocks": [
          {
            "ln_1": {"g": scale, "b": shift},
            "attn": {...},
            "ln_2": {...},
            "mlp": {...}
          },
          ...
        ],
        "ln_f": final_layer_norm_weights
      }
    """
    │
    ├─► Load embeddings
    │   gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])
    │   gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    │
    ├─► Load each transformer block
    │   for i, block in enumerate(gpt.trf_blocks):
    │       │
    │       ├─► Layer norm 1
    │       │   block.norm1.scale = assign(
    │       │       block.norm1.scale,
    │       │       params["blocks"][i]["ln_1"]["g"]
    │       │   )
    │       │   block.norm1.shift = assign(
    │       │       block.norm1.shift,
    │       │       params["blocks"][i]["ln_1"]["b"]
    │       │   )
    │       │
    │       ├─► Attention weights
    │       │   q, k, v weights combined in GPT-2
    │       │   Need to split them:
    │       │   qkv_weight = params["blocks"][i]["attn"]["c_attn"]["w"]
    │       │   q_w, k_w, v_w = split(qkv_weight, 3)
    │       │   block.att.W_query.weight = assign(q_w)
    │       │   block.att.W_key.weight = assign(k_w)
    │       │   block.att.W_value.weight = assign(v_w)
    │       │
    │       ├─► Output projection
    │       │   block.att.out_proj.weight = assign(
    │       │       params["blocks"][i]["attn"]["c_proj"]["w"]
    │       │   )
    │       │
    │       ├─► Layer norm 2
    │       │   (Similar to layer norm 1)
    │       │
    │       └─► Feed-forward network
    │           block.ff.layers[0].weight = assign(
    │               params["blocks"][i]["mlp"]["c_fc"]["w"]
    │           )
    │           block.ff.layers[2].weight = assign(
    │               params["blocks"][i]["mlp"]["c_proj"]["w"]
    │           )
    │
    └─► Load final layer norm
        gpt.final_norm.scale = assign(params["ln_f"]["g"])
        gpt.final_norm.shift = assign(params["ln_f"]["b"])
```

---

## Training Progress Monitoring

```
┌────────────────────────────────────────────────────────────────────────┐
│  TRACKING TRAINING PROGRESS                                           │
└────────────────────────────────────────────────────────────────────────┘

def train_model_simple(model, train_loader, val_loader, optimizer,
                       device, num_epochs, eval_freq, eval_iter,
                       start_context, tokenizer):
    │
    ├─► Initialize tracking lists
    │   train_losses = []
    │   val_losses = []
    │   track_tokens_seen = []
    │   tokens_seen = 0
    │   global_step = -1
    │
    ├─► Main training loop
    │   for epoch in range(num_epochs):
    │       model.train()
    │
    │       for input_batch, target_batch in train_loader:
    │           │
    │           ├─► Training step
    │           │   optimizer.zero_grad()
    │           │   loss = calc_loss_batch(input_batch, target_batch, model, device)
    │           │   loss.backward()
    │           │   optimizer.step()
    │           │
    │           ├─► Update counters
    │           │   tokens_seen += input_batch.numel()
    │           │   global_step += 1
    │           │
    │           └─► Periodic evaluation
    │               if global_step % eval_freq == 0:
    │                   train_loss, val_loss = evaluate_model(
    │                       model, train_loader, val_loader, device, eval_iter
    │                   )
    │                   train_losses.append(train_loss)
    │                   val_losses.append(val_loss)
    │                   track_tokens_seen.append(tokens_seen)
    │                   print(f"Ep {epoch+1} (Step {global_step:06d}): "
    │                         f"Train loss {train_loss:.3f}, "
    │                         f"Val loss {val_loss:.3f}")
    │
    │       # Generate sample after each epoch
    │       generate_and_print_sample(model, tokenizer, device, start_context)
    │
    └─► Return training history
        return train_losses, val_losses, track_tokens_seen


Example output:
───────────────────────────────────────────────────────────────────────
Ep 1 (Step 000005): Train loss 9.781, Val loss 9.933
Ep 1 (Step 000010): Train loss 8.111, Val loss 8.339
Every effort moves you toward the goal of the project.

Ep 2 (Step 000015): Train loss 6.661, Val loss 7.048
Ep 2 (Step 000020): Train loss 5.802, Val loss 6.589
Every effort moves you closer to your final destination.

Ep 3 (Step 000025): Train loss 5.333, Val loss 6.200
...
```

---

## Model Checkpointing

```python
# Save model checkpoint
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "tokens_seen": tokens_seen
}, "model_checkpoint.pt")

# Load model checkpoint
checkpoint = torch.load("model_checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
train_losses = checkpoint["train_losses"]
val_losses = checkpoint["val_losses"]
tokens_seen = checkpoint["tokens_seen"]
```

---

## Complete Training Example

```python
# Configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,      # Reduced for training
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

TRAINING_CONFIG = {
    "learning_rate": 5e-4,
    "weight_decay": 0.1,
    "batch_size": 2,
    "num_epochs": 10
}

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=TRAINING_CONFIG["learning_rate"],
    weight_decay=TRAINING_CONFIG["weight_decay"]
)

# Load data
with open("the-verdict.txt", "r") as f:
    text_data = f.read()

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))

train_loader = create_dataloader_v1(
    text_data[:split_idx],
    batch_size=TRAINING_CONFIG["batch_size"],
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True
)

val_loader = create_dataloader_v1(
    text_data[split_idx:],
    batch_size=TRAINING_CONFIG["batch_size"],
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False
)

# Train
tokenizer = tiktoken.get_encoding("gpt2")

train_losses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=TRAINING_CONFIG["num_epochs"],
    eval_freq=5,
    eval_iter=1,
    start_context="Every effort moves you",
    tokenizer=tokenizer
)

# Plot losses
import matplotlib.pyplot as plt
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
plt.savefig("training_losses.pdf")

# Save model
torch.save(model.state_dict(), "model.pth")

# Generate text
model.eval()
context = "The quick brown fox"
encoded = text_to_token_ids(context, tokenizer)
encoded_tensor = encoded.to(device)

with torch.no_grad():
    token_ids = generate(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=50,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=0.7,
        top_k=25
    )

decoded_text = token_ids_to_text(token_ids, tokenizer)
print(decoded_text)
```

---

## Key Training Metrics

```
┌────────────────────────────────────────────────────────────────────────┐
│  MONITORING TRAINING HEALTH                                           │
└────────────────────────────────────────────────────────────────────────┘

Good Training Signs:
  ✓ Train loss steadily decreasing
  ✓ Val loss following train loss (not diverging)
  ✓ Generated text improving over epochs
  ✓ Loss not fluctuating wildly

Warning Signs:
  ✗ Val loss increasing while train loss decreasing → Overfitting
  ✗ Both losses stuck at high value → Learning rate too low
  ✗ Loss becomes NaN or inf → Learning rate too high or numerical instability
  ✗ Val loss much higher than train loss → Dataset too small

Typical Loss Values:
  Random initialization:  ~10.5  (log(50257) ≈ 10.82)
  After 1 epoch:          ~6-8
  After 10 epochs:        ~4-6
  Well-trained (small):   ~3-4
  GPT-2 (full pretraining): ~2-3
```

---

## Optimization Details

**AdamW Optimizer:**
- Adaptive learning rates per parameter
- Momentum for smoother updates
- Weight decay for regularization
- Default betas: (0.9, 0.999)
- Epsilon: 1e-8

**Learning Rate:**
- Typical range: 1e-4 to 5e-4
- Too high: Training unstable, loss spikes
- Too low: Training too slow, may not converge

**Weight Decay:**
- L2 regularization on weights
- Typical: 0.1
- Prevents overfitting

**Batch Size:**
- Smaller batch (2-8): More gradient noise, slower but generalizes better
- Larger batch (32-128): Faster training, less noise, may overfit

---

## Code Location

- **Main notebook**: `ch05/01_main-chapter-code/ch05.ipynb`
- **Training script**: `ch05/01_main-chapter-code/gpt_train.py`
- **Generation script**: `ch05/01_main-chapter-code/gpt_generate.py`
- **Weight download**: `ch05/01_main-chapter-code/gpt_download.py`
- **Previous chapters**: `ch05/01_main-chapter-code/previous_chapters.py`

---

## Next Steps

After completing Chapter 5, you'll have:
- ✅ Trained a GPT model from scratch
- ✅ Implemented training and evaluation loops
- ✅ Loaded pretrained GPT-2 weights
- ✅ Generated text with temperature and top-k sampling
- ✅ Monitored training progress with loss curves

**Ready for Chapter 6**: Finetuning for classification! 🚀
