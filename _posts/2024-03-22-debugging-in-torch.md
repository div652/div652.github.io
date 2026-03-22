---
layout: post
title: "Debugging in Torch"
date: 2024-03-22 10:00:00
description: Practical tips for debugging PyTorch models like a software engineer.
tags: [pytorch, debugging, tech-notes]
categories: [tech-notes]
---

Debugging deep learning models can be frustrating because errors are often silent or opaque. Here are four practices to debug PyTorch models like a software engineer.

---

### 1. The Best Practice: Strict Shape Assertions

Don't trust PyTorch to align your tensors. Force it to crash if the dimensions aren't exactly what your mental model expects. Do this at the start of every major block.

```python
def forward(self, x):
    B, T, C = x.shape
    assert B == self.expected_batch_size, f"Expected Batch {self.expected_batch_size}, got {B}"

    # After a complex attention operation
    out = self.attention(x)
    assert out.shape == (B, T, self.head_size), "Attention output dimension mismatch!"
    return out
```

**Why this is great:** It acts as both a runtime check and live documentation for anyone reading your code.

---

### 2. The God-Tier Tool: `einops`

If there is one library you adopt as a pure CS student, make it **Einops** (Einstein Operations). It completely replaces `view`, `reshape`, and `transpose` with explicit, self-documenting strings. It strictly refuses to guess dimensions.

```python
from einops import rearrange

# Standard PyTorch (Hard to read, easy to accidentally broadcast)
x = x.view(B, T, num_heads, head_size).transpose(1, 2)

# Einops (Explicit, strict, and readable)
x = rearrange(x, 'batch time (heads head_size) -> batch heads time head_size', heads=8)
```

If the tensor `x` doesn't mathematically divide perfectly into `heads=8`, `einops` will throw a loud error instead of silently broadcasting.

---

### 3. Inspecting without Printing: Forward Hooks

If you want to know **what** the data is doing inside a massive `nn.Sequential` block without editing the `forward` function to add `print()` statements, you use **Hooks**. Hooks allow you to attach a listener to any node in the model.

```python
# Define a listener
def check_nan_hook(module, input, output):
    if torch.isnan(output).any():
        print(f"CRITICAL: NaN detected in layer {module.__class__.__name__}")

# Attach it to a specific layer from the outside
my_model.transformer_blocks[3].register_forward_hook(check_nan_hook)
```

---

### 4. Visualizing the DAG: `torchviz`

To ensure your computational graph is connected correctly (and to verify that `requires_grad=True` hasn't been accidentally lost, breaking your backpropagation), you can render the actual Abstract Syntax Tree that the autograd engine sees.

```python
from torchviz import make_dot

# Run a dummy batch
logits, loss = model(dummy_x, dummy_y)

# Generate a PDF diagram of the entire mathematical graph
make_dot(loss, params=dict(model.named_parameters())).render("model_graph", format="pdf")
```
