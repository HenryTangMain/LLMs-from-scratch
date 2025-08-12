== Types of parallelism ==

Data Parallelism (DP): Replicating the entire model on multiple devices and splitting the data across them.

Pros: Simple to implement.
Cons: Limited by the largest model that fits on a single device.
Model Parallelism (MP): Splitting the model's parameters across multiple devices.

Pros: Allows training of larger models.
Cons: Communication overhead between devices can slow down training.
Pipeline Parallelism (PP): Dividing the model into sequential stages and feeding micro-batches through the pipeline.

Pros: Balances memory and compute loads.
Cons: Pipeline bubbles (idle times) can reduce efficiency.
Tensor Parallelism (TP): Splitting individual tensors (weights) across devices.

Pros: Enables fine-grained parallelism.
Cons: Increased complexity in implementation.
Expert Parallelism (EP): Distributing expert layers (like in Mixture of Experts models) across devices.

Pros: Scales specific parts of the model.
Cons: May require specialized architectures.
Context Parallelism (CP): Dividing sequences into smaller chunks for parallel processing.

Pros: Optimizes memory usage for long sequences.
Cons: May introduce dependencies that need careful handling.
