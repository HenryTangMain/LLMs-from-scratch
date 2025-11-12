# Chapter 7: Finetuning to Follow Instructions - Call Flow Diagram

This document provides a detailed call flow diagram for instruction finetuning in Chapter 7.

## Overview

Chapter 7 demonstrates how to finetune a pretrained language model to follow instructions, similar to ChatGPT. It covers:

1. **Instruction Dataset Format** - Structured instruction-response pairs
2. **Custom Collate Function** - Variable-length sequence handling
3. **Masked Loss** - Training only on responses
4. **Instruction Finetuning** - Training to follow diverse instructions
5. **Model Evaluation** - Using external tools (Ollama) for evaluation
6. **Inference** - Generating responses to new instructions

---

## Instruction Format

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      INSTRUCTION FORMAT                                 │
└─────────────────────────────────────────────────────────────────────────┘

Standard Alpaca-style instruction format:

{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}

Formatted as:
──────────────────────────────────────────────────────────────────────────
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction:
What is the capital of France?

### Response:
The capital of France is Paris.
──────────────────────────────────────────────────────────────────────────


With additional input context:

{
  "instruction": "Summarize the following text",
  "input": "The quick brown fox jumps over the lazy dog. This sentence
            contains every letter of the English alphabet.",
  "output": "A pangram sentence demonstrating all alphabet letters."
}

Formatted as:
──────────────────────────────────────────────────────────────────────────
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction:
Summarize the following text

### Input:
The quick brown fox jumps over the lazy dog. This sentence contains
every letter of the English alphabet.

### Response:
A pangram sentence demonstrating all alphabet letters.
──────────────────────────────────────────────────────────────────────────


Why this format?
  ✓ Clear structure for model to learn
  ✓ Separates instruction, context, and response
  ✓ Standard format used by many instruction-tuned models
  ✓ Easy for humans to read and create
```

---

## Complete Instruction Finetuning Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│            CH07: FINETUNING TO FOLLOW INSTRUCTIONS                      │
│                 Instruction Finetuning Flow                             │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATA PREPARATION                                              │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├─► Load instruction dataset (JSON format)
  │   with open("instruction-data.json", "r") as f:
  │       data = json.load(f)
  │   │
  │   │   [
  │   │     {
  │   │       "instruction": "Identify the verb in the sentence",
  │   │       "input": "The cat sleeps on the couch.",
  │   │       "output": "sleeps"
  │   │     },
  │   │     {
  │   │       "instruction": "Convert to uppercase",
  │   │       "input": "hello world",
  │   │       "output": "HELLO WORLD"
  │   │     },
  │   │     ...
  │   │   ]
  │   │
  │   └─► Typically 1,000 - 100,000 examples
  │
  ├─► Split into train/validation/test
  │   train_portion = int(len(data) * 0.85)
  │   test_portion = int(len(data) * 0.1)
  │   │
  │   train_data = data[:train_portion]
  │   test_data = data[train_portion:train_portion + test_portion]
  │   val_data = data[train_portion + test_portion:]
  │
  └─► Format each example
      def format_input(entry):
          instruction_text = (
              f"Below is an instruction that describes a task. "
              f"Write a response that appropriately completes the request."
              f"\n\n### Instruction:\n{entry['instruction']}"
          )
          input_text = (
              f"\n\n### Input:\n{entry['input']}"
              if entry["input"] else ""
          )
          return instruction_text + input_text


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 2: INSTRUCTION DATASET CLASS                                    │
└────────────────────────────────────────────────────────────────────────┘

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize all texts
        self.encoded_texts = []
        for entry in data:
            # Format instruction + input
            instruction_plus_input = format_input(entry)

            # Add response
            response_text = f"\n\n### Response:\n{entry['output']}"

            # Full text = instruction + input + response
            full_text = instruction_plus_input + response_text

            # Tokenize
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


Example:
────────────────────────────────────────────────────────────────────────
Entry: {
  "instruction": "What is 2+2?",
  "input": "",
  "output": "4"
}

Formatted text:
"Below is an instruction that describes a task. Write a response that
appropriately completes the request.\n\n### Instruction:\nWhat is 2+2?
\n\n### Response:\n4"

Tokenized: [token_ids...]
Length: ~50 tokens (varies by entry)


┌────────────────────────────────────────────────────────────────────────┐
│  STEP 3: CUSTOM COLLATE FUNCTION (Variable Length Handling)          │
└────────────────────────────────────────────────────────────────────────┘

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                      allowed_max_length=None, device="cpu"):
    """
    Custom collate function to handle variable-length sequences.

    Key features:
      - Pads sequences to longest in batch
      - Masks padding tokens in loss calculation
      - Creates input/target pairs (shifted by 1)
    """
    │
    ├─► Find longest sequence in batch
    │   batch_max_length = max(len(item) + 1 for item in batch)
    │   │
    │   │   Example batch:
    │   │   item 0: [1, 2, 3, 4, 5]           (length 5)
    │   │   item 1: [10, 11, 12]              (length 3)
    │   │   item 2: [20, 21, 22, 23, 24, 25]  (length 6)
    │   │   │
    │   │   └─► batch_max_length = 6 + 1 = 7
    │   │
    │   └─► +1 for <|endoftext|> token
    │
    ├─► Process each item
    │   inputs_lst, targets_lst = [], []
    │   │
    │   for item in batch:
    │       │
    │       ├─► Add <|endoftext|> token
    │       │   new_item = item + [pad_token_id]
    │       │
    │       ├─► Pad to batch_max_length
    │       │   padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
    │       │   │
    │       │   │   Example for item [1, 2, 3]:
    │       │   │   new_item: [1, 2, 3, 50256]
    │       │   │   padded:   [1, 2, 3, 50256, 50256, 50256, 50256]
    │       │   │              └─ len=7 (batch_max_length)
    │       │
    │       ├─► Create inputs and targets (shifted by 1)
    │       │   inputs = padded[:-1]   # All but last token
    │       │   targets = padded[1:]   # All but first token (shifted right)
    │       │   │
    │       │   │   padded:  [1, 2, 3, 50256, 50256, 50256, 50256]
    │       │   │   inputs:  [1, 2, 3, 50256, 50256, 50256]
    │       │   │   targets: [2, 3, 50256, 50256, 50256, 50256]
    │       │
    │       └─► Mask padding tokens (except first)
    │           mask = (targets == pad_token_id)
    │           indices = torch.nonzero(mask).squeeze()
    │           if indices.numel() > 1:
    │               targets[indices[1:]] = ignore_index
    │           │
    │           │   Before masking:
    │           │   targets: [2, 3, 50256, 50256, 50256, 50256]
    │           │                       ↑     └──── mask these ────┘
    │           │                   keep first
    │           │
    │           │   After masking:
    │           │   targets: [2, 3, 50256, -100, -100, -100]
    │           │                       ↑     └─ ignored in loss ─┘
    │           │                    predict <|endoftext|>
    │           │
    │           └─► ignore_index (-100) tells loss function to skip these
    │
    └─► Stack into tensors
        inputs_tensor = torch.stack(inputs_lst).to(device)
        targets_tensor = torch.stack(targets_lst).to(device)
        │
        └─► Return (inputs, targets)


Why mask padding?
  - Don't want model to learn to predict padding tokens
  - Focus loss on actual content
  - ignore_index=-100 is PyTorch convention (ignored by cross_entropy)


┌────────────────────────────────────────────────────────────────────────┐
│  MASKED LOSS VISUALIZATION                                            │
└────────────────────────────────────────────────────────────────────────┘

Example sequence after processing:

Inputs:  [Below, is, an, instruction, ..., Response, :, 4, <|endoftext|>, <pad>]
Targets: [is, an, instruction, ..., Response, :, 4, <|endoftext|>, <pad>, <pad>]

Loss mask (what we compute loss on):

Position:    0     1    2      3         ...    N-3  N-2  N-1   N     N+1
Inputs:   [Below, is, an, instruction, ..., Response, :,  4, <|end|>, <pad>]
Targets:  [  is,  an, instruction, ..., Response, :,  4, <|end|>, <pad>, <pad>]
Compute    ✓     ✓    ✓      ✓        ...    ✓     ✓   ✓     ✓      ✗     ✗
loss?                                                              │     └─ masked
                                                                   └─── masked

All actual content: compute loss
Padding tokens (except first <|endoftext|>): masked


Optional: Train only on response
──────────────────────────────────────────────────────────────────────────
Some implementations only compute loss on the response portion:

Full text: [instruction_tokens, response_tokens, <|end|>, <pad>, <pad>]
Mask:      [      XXXX (ignore)      ,     ✓ (train)   ,   ✓  ,  ✗  ,  ✗  ]

This focuses learning on generating good responses, not memorizing instructions.
```

---

## Training Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│  INSTRUCTION FINETUNING TRAINING LOOP                                 │
└────────────────────────────────────────────────────────────────────────┘

def train_model_simple(model, train_loader, val_loader, optimizer,
                       device, num_epochs, eval_freq, eval_iter,
                       start_context, tokenizer):
    │
    └─► Same as Ch05 training loop, but:
        │
        ├─► Uses custom_collate_fn for data loading
        │   train_loader = DataLoader(
        │       train_dataset,
        │       batch_size=8,
        │       shuffle=True,
        │       collate_fn=partial(custom_collate_fn, device=device)
        │   )
        │
        ├─► Loss automatically handles ignore_index=-100
        │   loss = F.cross_entropy(
        │       logits.flatten(0, 1),
        │       targets.flatten(),
        │       ignore_index=-100  # ← Skip masked tokens
        │   )
        │
        └─► Evaluation generates instruction-following responses
            generate_and_print_sample(
                model, tokenizer, device,
                "### Instruction:\nWhat is 2+2?\n\n### Response:\n"
            )


Example training output:
────────────────────────────────────────────────────────────────────────
Ep 1 (Step 000010): Train loss 2.831, Val loss 2.756
Sample: "### Instruction:\nWhat is 2+2?\n\n### Response:\n4"

Ep 1 (Step 000020): Train loss 2.345, Val loss 2.298
Sample: "### Instruction:\nWhat is the capital of France?\n\n### Response:\nParis"

Ep 2 (Step 000030): Train loss 1.876, Val loss 1.842
Sample: "### Instruction:\nExplain photosynthesis\n\n### Response:\n
         Photosynthesis is the process plants use to convert sunlight into energy..."

Loss decreases → Model learning to follow instructions!
```

---

## Model Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  INSTRUCTION FINETUNING MODEL                                         │
└────────────────────────────────────────────────────────────────────────┘

SAME GPT model as Ch04/Ch05:
  - No architectural changes
  - Still predicts next token
  - Output head: Linear(emb_dim, vocab_size)

DIFFERENCE is in TRAINING DATA:
  Ch05 (Pretraining):
    - Raw text: "The cat sat on the mat"
    - Learn: General language patterns

  Ch07 (Instruction Finetuning):
    - Structured: "### Instruction:\n...\n\n### Response:\n..."
    - Learn: To follow instructions and generate appropriate responses

Think of it as:
  Pretraining = Learning to write English
  Instruction finetuning = Learning to be a helpful assistant
```

---

## Inference (Generating Responses)

```python
def generate_instruction_response(model, tokenizer, instruction, input_text="",
                                 max_new_tokens=256, temperature=0.7, top_k=50):
    """
    Generate response to an instruction.
    """
    # Format the prompt
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}"""

    if input_text:
        prompt += f"\n\n### Input:\n{input_text}"

    prompt += "\n\n### Response:\n"

    # Tokenize
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    # Generate
    model.eval()
    with torch.no_grad():
        output_ids = generate(
            model=model,
            idx=input_tensor,
            max_new_tokens=max_new_tokens,
            context_size=model.pos_emb.weight.shape[0],
            temperature=temperature,
            top_k=top_k
        )

    # Decode
    output_text = tokenizer.decode(output_ids[0].tolist())

    # Extract response (after "### Response:\n")
    response_start = output_text.find("### Response:\n") + len("### Response:\n")
    response = output_text[response_start:].strip()

    # Optional: Stop at next "###" or <|endoftext|>
    if "###" in response:
        response = response[:response.find("###")].strip()

    return response


# Example usage
instruction = "What are the benefits of regular exercise?"
response = generate_instruction_response(
    model, tokenizer, instruction,
    max_new_tokens=200,
    temperature=0.7,
    top_k=50
)
print(response)

# Output might be:
# "Regular exercise has numerous benefits including improved cardiovascular
#  health, better mental well-being, increased energy levels, stronger muscles
#  and bones, and better sleep quality. It can also help with weight management
#  and reduce the risk of chronic diseases."
```

---

## Evaluation with Ollama

```python
def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    """
    Query Ollama for evaluation.

    Used to evaluate model responses using a stronger model (like Llama 3).
    """
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": 123, "temperature": 0}
    }

    response = requests.post(url, json=data)
    response_json = response.json()

    # Collect response
    full_response = ""
    for line in response.text.strip().split("\n"):
        json_obj = json.loads(line)
        if "message" in json_obj:
            full_response += json_obj["message"]["content"]

    return full_response


def evaluate_response(instruction, model_response):
    """
    Evaluate if response correctly follows instruction.
    """
    eval_prompt = f"""
Given the instruction:
{instruction}

And the response:
{model_response}

Score the response quality from 0-100 based on:
- Correctness
- Relevance
- Completeness
- Clarity

Provide ONLY a numeric score (0-100).
"""

    score_text = query_model(eval_prompt, model="llama3")

    # Extract numeric score
    try:
        score = int(score_text.strip())
        return score
    except:
        return None


# Example evaluation
instruction = "What is 2+2?"
model_response = "4"
score = evaluate_response(instruction, model_response)
print(f"Score: {score}/100")  # Expected: 100


instruction = "Explain quantum computing"
model_response = "Quantum computing uses quantum mechanics principles..."
score = evaluate_response(instruction, model_response)
print(f"Score: {score}/100")  # Expected: 70-90
```

---

## Key Differences from Previous Chapters

| Aspect | Ch05 (Pretraining) | Ch06 (Classification) | Ch07 (Instruction) |
|--------|-------------------|----------------------|--------------------|
| **Data format** | Raw text | Labeled examples | Instruction-response pairs |
| **Output** | Next token (all positions) | Class label (last token) | Next token (all positions) |
| **Task** | Language modeling | Classification | Instruction following |
| **Loss** | All tokens | Last token only | Instruction + response (or response only) |
| **Evaluation** | Perplexity | Accuracy | Human eval / LLM-as-judge |
| **Model head** | vocab_size | num_classes | vocab_size (same as pretraining) |
| **Training epochs** | Many (100+) | Few (5-10) | Few (2-5) |
| **Dataset size** | Very large (GB) | Medium (1k-100k) | Small-medium (1k-100k) |

---

## Best Practices

```
┌────────────────────────────────────────────────────────────────────────┐
│  INSTRUCTION FINETUNING BEST PRACTICES                                │
└────────────────────────────────────────────────────────────────────────┘

Data Quality:
  ✓ High-quality instruction-response pairs
  ✓ Diverse instruction types (Q&A, summarization, rewriting, etc.)
  ✓ Varied difficulty levels
  ✓ Clear, unambiguous instructions
  ✗ Avoid contradictory examples
  ✗ Remove low-quality or nonsensical pairs

Dataset Size:
  - Minimum: ~1,000 high-quality examples
  - Good: 10,000 - 50,000 examples
  - Excellent: 100,000+ examples
  - Diminishing returns beyond 1M examples without quality

Hyperparameters:
  - Learning rate: 5e-6 to 5e-5 (lower than pretraining)
  - Epochs: 2-5 (more risks overfitting)
  - Batch size: 4-32
  - Max sequence length: 512-2048
  - Warmup steps: 100-500

Training Strategy:
  ✓ Start from pretrained model
  ✓ Use lower learning rate than pretraining
  ✓ Train for fewer epochs
  ✓ Monitor validation loss carefully
  ✓ Early stopping if validation loss increases
  ✗ Don't train from scratch (too expensive)

Evaluation:
  ✓ Human evaluation (gold standard)
  ✓ LLM-as-judge (Ollama, GPT-4)
  ✓ Task-specific metrics
  ✓ Qualitative inspection of outputs
  ✗ Perplexity alone (not sufficient)

Common Issues:
  - Overfitting: Model memorizes training examples
    → Solution: More data, fewer epochs, regularization
  - Hallucination: Model makes up facts
    → Solution: Better training data, fact-checking
  - Instruction confusion: Doesn't follow format
    → Solution: Consistent format, more examples
  - Catastrophic forgetting: Loses pretraining knowledge
    → Solution: Lower learning rate, mixing in pretraining data
```

---

## Complete Example Script

```python
# ============================================
# SETUP
# ============================================
import json
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
import tiktoken

from previous_chapters import GPTModel, load_weights_into_gpt, generate
from gpt_download import download_and_load_gpt2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# LOAD PRETRAINED MODEL
# ============================================
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12
}

model = GPTModel(BASE_CONFIG)
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
load_weights_into_gpt(model, params)
model.to(device)

# ============================================
# LOAD DATA
# ============================================
with open("instruction-data.json", "r") as f:
    data = json.load(f)

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

# ============================================
# CREATE DATASETS
# ============================================
tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = InstructionDataset(train_data, tokenizer)
val_dataset = InstructionDataset(val_data, tokenizer)

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=customized_collate_fn,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=customized_collate_fn,
    drop_last=False
)

# ============================================
# TRAIN
# ============================================
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 2
train_losses, val_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="### Instruction:\nWhat is 1+1?\n\n### Response:\n",
    tokenizer=tokenizer
)

# ============================================
# SAVE MODEL
# ============================================
torch.save(model.state_dict(), "instruction_finetuned_model.pth")

# ============================================
# TEST INFERENCE
# ============================================
test_instructions = [
    "What is the capital of Germany?",
    "Explain photosynthesis in simple terms",
    "Convert this to uppercase: hello world"
]

for instruction in test_instructions:
    response = generate_instruction_response(
        model, tokenizer, instruction,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50
    )
    print(f"\nInstruction: {instruction}")
    print(f"Response: {response}")
```

---

## Code Location

- **Main notebook**: `ch07/01_main-chapter-code/ch07.ipynb`
- **Instruction finetuning script**: `ch07/01_main-chapter-code/gpt_instruction_finetuning.py`
- **Ollama evaluation**: `ch07/01_main-chapter-code/ollama_evaluate.py`
- **Previous chapters**: `ch07/01_main-chapter-code/previous_chapters.py`
- **Instruction dataset**: `ch07/01_main-chapter-code/instruction-data.json`

---

## Next Steps

After completing Chapter 7, you'll have:
- ✅ Finetuned a model to follow instructions
- ✅ Implemented custom data handling for variable-length sequences
- ✅ Used masked loss for targeted training
- ✅ Evaluated instruction-following quality
- ✅ Built a ChatGPT-like assistant

**You've completed the full LLM pipeline!** 🎉

From raw text → pretrained model → task-specific finetuning → instruction following!
