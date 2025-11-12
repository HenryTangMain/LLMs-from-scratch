# Chapter 6: Finetuning for Text Classification - Call Flow Diagram

This document provides a detailed call flow diagram for finetuning a GPT model for text classification in Chapter 6.

## Overview

Chapter 6 demonstrates how to adapt a pretrained language model for a specific classification task (spam detection). It covers:

1. **Classification Head** - Adding task-specific output layer
2. **Dataset Preparation** - Loading and preprocessing labeled data
3. **Freezing Layers** - Selective parameter training
4. **Classification Training** - Modified training loop for classification
5. **Accuracy Evaluation** - Computing classification metrics
6. **Full Model Finetuning** - Training all parameters

---

## Classification vs Language Modeling

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  LANGUAGE MODEL vs CLASSIFIER                           │
└─────────────────────────────────────────────────────────────────────────┘

LANGUAGE MODEL (Ch05):
──────────────────────────────────────────────────────────────────────────
  Input:  "The cat sat"
          [token_0, token_1, token_2]
          ↓
  Model:  GPT (all layers)
          ↓
  Output: Logits for EACH position
          [vocab_size scores] for token_0 → predicts token_1
          [vocab_size scores] for token_1 → predicts token_2
          [vocab_size scores] for token_2 → predicts token_3
          ↓
  Loss:   Cross-entropy for next-token prediction
          Average across all positions

  Purpose: Generate coherent text


CLASSIFIER (Ch06):
──────────────────────────────────────────────────────────────────────────
  Input:  "Buy now! Limited offer!"
          [token_0, token_1, ..., token_n]
          ↓
  Model:  GPT (all layers) + Classification head
          ↓
  Output: Logits for LAST position only
          [num_classes scores]  ← e.g., [spam_score, ham_score]
          ↓
  Loss:   Cross-entropy for classification
          Single prediction per input

  Purpose: Classify entire text into categories
```

---

## Complete Classification Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│               CH06: FINETUNING FOR CLASSIFICATION                       │
│                   Classification Pipeline Flow                          │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA PREPARATION                                              │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► Download spam dataset
  │   URL: SMS Spam Collection
  │   Format: "ham/spam \t message text"
  │
  ├─► Load into DataFrame
  │   df = pd.read_csv("sms_spam_collection.tsv", sep="\t")
  │   │
  │   │   Label    Text
  │   │   ─────    ────────────────────────────────
  │   │   ham      "How are you doing today?"
  │   │   spam     "WINNER! Claim your prize now!"
  │   │   ham      "See you at the meeting"
  │   │   spam     "Call now for free offer!!!"
  │   │   ...
  │
  ├─► Balance dataset (equal spam/ham)
  │   num_spam = df[df["Label"] == "spam"].shape[0]
  │   ham_subset = df[df["Label"] == "ham"].sample(num_spam)
  │   balanced_df = pd.concat([ham_subset, spam_df])
  │   │
  │   └─► Prevents class imbalance bias
  │
  ├─► Encode labels
  │   balanced_df["Label"] = balanced_df["Label"].map(
  │       {"ham": 0, "spam": 1}
  │   )
  │
  └─► Split into train/val/test
      train_df, val_df, test_df = random_split(
          balanced_df,
          train_frac=0.7,
          validation_frac=0.1
      )
      # test_frac = 0.2 (remaining)


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 2: DATASET CLASS FOR CLASSIFICATION                             │
└────────────────────────────────────────────────────────────────────────┘

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        │
        ├─► Load data
        │   self.data = pd.read_csv(csv_file)
        │
        ├─► Tokenize all texts
        │   self.encoded_texts = [
        │       tokenizer.encode(text) for text in self.data["Text"]
        │   ]
        │
        ├─► Determine max_length
        │   if max_length is None:
        │       self.max_length = max(len(text) for text in self.encoded_texts)
        │   else:
        │       self.max_length = max_length
        │       # Truncate sequences longer than max_length
        │       self.encoded_texts = [
        │           text[:max_length] for text in self.encoded_texts
        │       ]
        │
        └─► Pad all sequences to max_length
            self.encoded_texts = [
                text + [pad_token_id] * (max_length - len(text))
                for text in self.encoded_texts
            ]

    def __getitem__(self, index):
        """Return (encoded_text, label) pair"""
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)


Example:
────────────────────────────────────────────────────────────────────────
Text: "WINNER! Call now!"
Tokens: [WIN, NER, !, Call, now, !]
Token IDs: [12345, 678, 0, 2345, 890, 0]

After padding (max_length=10):
  [12345, 678, 0, 2345, 890, 0, 50256, 50256, 50256, 50256]
   └────────── actual ──────────┘ └────── padding ────────┘

Label: 1 (spam)
```

---

## Model Architecture Modification

```
┌────────────────────────────────────────────────────────────────────────┐
│  CLASSIFICATION HEAD                                                   │
└────────────────────────────────────────────────────────────────────────┘

ORIGINAL GPT MODEL (Language Modeling):
──────────────────────────────────────────────────────────────────────────
  Input tokens → Embeddings → Transformer Blocks → Final LayerNorm
                                                          ↓
                                                    Linear(emb_dim, vocab_size)
                                                          ↓
                                                    Logits [batch, seq_len, 50257]


MODIFIED FOR CLASSIFICATION:
──────────────────────────────────────────────────────────────────────────
  Input tokens → Embeddings → Transformer Blocks → Final LayerNorm
                                                          ↓
                            Select LAST token representation ←─┐
                                    x[:, -1, :]                │
                                          ↓                    │
                            Linear(emb_dim, num_classes) ←───┐│
                                          ↓                   ││
                            Logits [batch, num_classes]      ││
                                                              ││
Why last token?                                               ││
  - Causal attention means last token "sees" all previous    ││
  - Last token has full context of entire sequence           ││
  - Common practice in GPT-based classification              ││
                                                              ││
  Input:  [CLS] [token1] [token2] ... [tokenN]               ││
           ↓       ↓        ↓           ↓                     ││
          attended → attended → attended → [LAST] ───────────┘│
                                            ↓                  │
                                  Use this for classification ─┘


MODEL FORWARD PASS:
──────────────────────────────────────────────────────────────────────────
def forward(self, in_idx):
    # Standard GPT forward pass
    batch_size, seq_len = in_idx.shape
    x = self.tok_emb(in_idx) + self.pos_emb(torch.arange(seq_len))
    x = self.drop_emb(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)

    # Classification: use LAST token only
    logits = self.out_head(x[:, -1, :])  # [batch, seq_len, emb_dim]
                                          #         ↓
                                          # [batch, emb_dim]
                                          #         ↓
                                          # [batch, num_classes]
    return logits
```

---

## Training Strategy: Freezing Layers

```
┌────────────────────────────────────────────────────────────────────────┐
│  LAYER FREEZING STRATEGY                                              │
└────────────────────────────────────────────────────────────────────────┘

STRATEGY 1: Freeze All Transformer Blocks
──────────────────────────────────────────────────────────────────────────
  Freeze: All transformer blocks (pretrained knowledge)
  Train:  Only classification head (task-specific)

  # Freeze transformer blocks
  for param in model.trf_blocks.parameters():
      param.requires_grad = False

  # Classification head remains trainable
  for param in model.out_head.parameters():
      param.requires_grad = True

  Advantages:
    ✓ Fast training (fewer parameters)
    ✓ Less risk of overfitting
    ✓ Works well with small datasets

  Disadvantages:
    ✗ Limited adaptation to new domain
    ✗ May underperform on very different tasks


STRATEGY 2: Freeze Lower Layers Only
──────────────────────────────────────────────────────────────────────────
  Freeze: First 6 transformer blocks (general features)
  Train:  Last 6 blocks + classification head (task adaptation)

  # Freeze first 6 blocks
  for block in model.trf_blocks[:6]:
      for param in block.parameters():
          param.requires_grad = False

  # Unfreeze last 6 blocks
  for block in model.trf_blocks[6:]:
      for param in block.parameters():
          param.requires_grad = True

  Advantages:
    ✓ Better task adaptation
    ✓ Still relatively fast
    ✓ Good balance

  Disadvantages:
    ✗ More parameters to tune
    ✗ Slightly higher risk of overfitting


STRATEGY 3: Train All Layers (Full Finetuning)
──────────────────────────────────────────────────────────────────────────
  Freeze: Nothing
  Train:  All parameters

  # Unfreeze everything
  for param in model.parameters():
      param.requires_grad = True

  Advantages:
    ✓ Maximum task adaptation
    ✓ Best potential performance
    ✓ Can adapt to very different domains

  Disadvantages:
    ✗ Slow training (many parameters)
    ✗ High risk of overfitting with small data
    ✗ Requires careful hyperparameter tuning


PARAMETER COUNT COMPARISON (GPT-2 124M):
──────────────────────────────────────────────────────────────────────────
  Strategy 1 (classification head only):     ~1.5K parameters
  Strategy 2 (last 6 blocks + head):        ~44M parameters
  Strategy 3 (all layers):                  ~124M parameters
```

---

## Classification Training Loop

```
┌────────────────────────────────────────────────────────────────────────┐
│  CLASSIFICATION TRAINING                                              │
└────────────────────────────────────────────────────────────────────────┘

def train_classifier_simple(model, train_loader, val_loader, optimizer,
                            device, num_epochs, eval_freq, eval_iter):
    │
    ├─► Initialize tracking
    │   train_losses, val_losses = [], []
    │   train_accs, val_accs = [], []
    │   examples_seen, global_step = 0, -1
    │
    └─► Main training loop
        for epoch in range(num_epochs):
            model.train()

            for input_batch, target_batch in train_loader:
                │
                ├─► Forward pass
                │   input_batch = input_batch.to(device)
                │   target_batch = target_batch.to(device)
                │   │
                │   │   input_batch: [batch_size, seq_len]
                │   │   target_batch: [batch_size]  ← Single label per input
                │   │
                │   logits = model(input_batch)
                │   │
                │   └─► logits: [batch_size, num_classes]
                │       Example: [8, 2]  (2 classes: ham, spam)
                │
                ├─► Compute loss
                │   loss = calc_loss_batch(
                │       input_batch, target_batch, model, device
                │   )
                │   │
                │   │   Inside calc_loss_batch:
                │   │   ─────────────────────────────────────────────
                │   │   logits = model(input_batch)[:, -1, :]
                │   │   loss = F.cross_entropy(logits, target_batch)
                │   │
                │   └─► Single classification loss (not per-token)
                │
                ├─► Backward pass
                │   optimizer.zero_grad()
                │   loss.backward()
                │   optimizer.step()
                │
                ├─► Update counters
                │   examples_seen += input_batch.shape[0]
                │   global_step += 1
                │
                └─► Periodic evaluation
                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(...)
                        train_acc = calc_accuracy_loader(train_loader, ...)
                        val_acc = calc_accuracy_loader(val_loader, ...)

                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        train_accs.append(train_acc)
                        val_accs.append(val_acc)

                        print(f"Ep {epoch+1} (Step {global_step:06d}): "
                              f"Train loss {train_loss:.3f}, "
                              f"Val loss {val_loss:.3f}, "
                              f"Train acc {train_acc:.2f}, "
                              f"Val acc {val_acc:.2f}")

        return train_losses, val_losses, train_accs, val_accs, examples_seen
```

---

## Accuracy Calculation

```
┌────────────────────────────────────────────────────────────────────────┐
│  COMPUTING CLASSIFICATION ACCURACY                                    │
└────────────────────────────────────────────────────────────────────────┘

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    Compute classification accuracy.

    Returns: Fraction of correct predictions
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        with torch.no_grad():
            # Get logits for last token
            logits = model(input_batch)[:, -1, :]
            # Shape: [batch_size, num_classes]

            # Get predicted class (highest score)
            predicted_labels = torch.argmax(logits, dim=-1)
            # Shape: [batch_size]

            # Compare with true labels
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples


Example:
────────────────────────────────────────────────────────────────────────
Batch size: 4

Logits: [[ 2.3, -1.2],  ← Text 1: ham score=2.3, spam score=-1.2
         [-0.5,  3.1],  ← Text 2: ham score=-0.5, spam score=3.1
         [ 1.8, -0.9],  ← Text 3: ham score=1.8, spam score=-0.9
         [-1.0,  2.7]]  ← Text 4: ham score=-1.0, spam score=2.7

Predicted: [0, 1, 0, 1]  ← argmax of each row
           (ham, spam, ham, spam)

Target:    [0, 1, 1, 1]  ← True labels
           (ham, spam, spam, spam)

Correct:   [✓, ✓, ✗, ✓]  ← 3 out of 4 correct

Accuracy: 3/4 = 0.75 = 75%
```

---

## Complete Classification Example

```python
# ============================================
# STEP 1: Load Pretrained Model
# ============================================
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt

# Base configuration
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

# Model sizes
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Choose model
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Load weights
settings, params = download_and_load_gpt2(
    model_size="355M",
    models_dir="gpt2"
)

# Initialize model
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()


# ============================================
# STEP 2: Modify for Classification
# ============================================
# Replace output head for classification
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ============================================
# STEP 3: Freeze Layers (Optional)
# ============================================
# Option 1: Freeze all transformer blocks
for param in model.trf_blocks.parameters():
    param.requires_grad = False

# Option 2: Freeze only first 6 blocks
# for block in model.trf_blocks[:6]:
#     for param in block.parameters():
#         param.requires_grad = False

# Classification head is always trainable
for param in model.out_head.parameters():
    param.requires_grad = True


# ============================================
# STEP 4: Prepare Data
# ============================================
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# Download and prepare spam dataset
download_and_unzip_spam_data(
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
    zip_path="sms_spam_collection.zip",
    extracted_path="sms_spam_collection",
    data_file_path="sms_spam_collection.tsv"
)

# Load and balance
df = pd.read_csv("sms_spam_collection.tsv", sep="\t", header=None, names=["Label", "Text"])
balanced_df = create_balanced_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# Split
train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)
train_df.to_csv("train.csv", index=None)
val_df.to_csv("val.csv", index=None)
test_df.to_csv("test.csv", index=None)

# Create datasets
train_dataset = SpamDataset("train.csv", tokenizer, max_length=120)
val_dataset = SpamDataset("val.csv", tokenizer, max_length=120)
test_dataset = SpamDataset("test.csv", tokenizer, max_length=120)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# ============================================
# STEP 5: Train
# ============================================
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=50,
    eval_iter=5
)


# ============================================
# STEP 6: Evaluate
# ============================================
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy:   {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy:       {test_accuracy*100:.2f}%")


# ============================================
# STEP 7: Inference on New Text
# ============================================
def classify_review(text, model, tokenizer, device, max_length=None):
    model.eval()

    # Tokenize and pad
    input_ids = tokenizer.encode(text)
    if max_length is not None:
        input_ids = input_ids[:max_length]

    # Create batch dimension
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Last token

    # Get predicted class
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "ham"

# Test on new examples
text_1 = "You are a winner! Call now to claim your prize!"
text_2 = "Hey, can we meet for lunch tomorrow?"

print(f"Text 1: {classify_review(text_1, model, tokenizer, device, max_length=120)}")
print(f"Text 2: {classify_review(text_2, model, tokenizer, device, max_length=120)}")
```

---

## Training Monitoring

```
Typical Output:
────────────────────────────────────────────────────────────────────────
Ep 1 (Step 000050): Train loss 0.433, Val loss 0.287, Train acc 0.78, Val acc 0.85
Ep 1 (Step 000100): Train loss 0.312, Val loss 0.235, Train acc 0.86, Val acc 0.89
Ep 2 (Step 000150): Train loss 0.256, Val loss 0.201, Train acc 0.90, Val acc 0.92
Ep 2 (Step 000200): Train loss 0.198, Val loss 0.178, Train acc 0.93, Val acc 0.94
Ep 3 (Step 000250): Train loss 0.167, Val loss 0.165, Train acc 0.95, Val acc 0.95
...

Final Results:
────────────────────────────────────────────────────────────────────────
Training accuracy:   97.20%
Validation accuracy: 95.50%
Test accuracy:       96.10%

Good signs:
  ✓ Validation accuracy close to training accuracy
  ✓ Test accuracy similar to validation accuracy
  ✓ High accuracy (>95%) on both classes

Warning signs:
  ✗ Train accuracy >> Val accuracy → Overfitting
  ✗ Very low accuracy → Model not learning or data issues
  ✗ One class accuracy much higher → Class imbalance
```

---

## Key Differences from Language Modeling

| Aspect | Language Modeling (Ch05) | Classification (Ch06) |
|--------|-------------------------|----------------------|
| **Task** | Predict next token | Classify entire sequence |
| **Output** | All token positions | Last token only |
| **Logits shape** | `[B, T, V]` | `[B, C]` |
| **Loss** | Average over all positions | Single loss per input |
| **Labels** | Next tokens (self-supervised) | External labels (supervised) |
| **Evaluation** | Perplexity, generation quality | Accuracy, F1, precision/recall |
| **Training data** | Unlabeled text | Labeled examples |
| **Dataset size** | Large (millions of tokens) | Small (thousands of examples) |

Legend: B=batch, T=sequence_length, V=vocab_size, C=num_classes

---

## Code Location

- **Main notebook**: `ch06/01_main-chapter-code/ch06.ipynb`
- **Classification script**: `ch06/01_main-chapter-code/gpt_class_finetune.py`
- **Weight download**: `ch06/01_main-chapter-code/gpt_download.py`
- **Previous chapters**: `ch06/01_main-chapter-code/previous_chapters.py`

---

## Next Steps

After completing Chapter 6, you'll have:
- ✅ Finetuned a GPT model for classification
- ✅ Learned layer freezing strategies
- ✅ Implemented classification-specific data loading
- ✅ Computed accuracy metrics
- ✅ Applied pretrained models to new tasks

**Ready for Chapter 7**: Instruction finetuning! 🚀
