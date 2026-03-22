---
layout: post
title: "Batching in Torch"
date: 2024-03-21 15:00:00
description: A CS student's guide to PyTorch batching rules.
tags: [pytorch, machine-learning, torch]
categories: [tech-notes]
---

In PyTorch, the `for` loop is written in C++ and hidden behind Python math operators. This makes the code fast, but it feels like "magic" because you can't see the loop.

The good news is: You do not need to memorize every operator. Almost every operation in PyTorch falls into one of **Three Golden Rules**. If you understand these three rules, you understand 99% of PyTorch batching.

[PyTorch Blitz Tutorial](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)

---

### Rule 1: The Broadcasting Theorem (Element-wise Ops)

**Applies to:** `+`, `-`, `*`, `/`, `==`, `masked_fill`, `torch.exp`, `torch.relu`

When you do math between two tensors of different shapes, PyTorch tries to implicitly write the `for` loops for you. It does this using **Broadcasting**.

#### The Algorithm (How PyTorch thinks):

1.  **Right-Alignment:** Align the shapes of the two tensors strictly to the right.
2.  **Left-Padding:** If one tensor has fewer dimensions, pad its left side with `1`s.
3.  **The Compatibility Test:** Compare the aligned dimensions. They are compatible ONLY if:
    - They are exactly equal.
    - One of them is `1` (PyTorch will "stretch" the `1` to match the other).

#### Example: Adding a Bias to a Sequence

You have a batch of sequences `(Batch=32, Time=10, Channels=64)` and a bias vector `(Channels=64)`.

```text
Tensor A (Batch): 32 x 10 x 64
Tensor B (Bias) :           64
--------------------------------------
Step 1 (Align right): 32 x 10 x 64
                               64
Step 2 (Pad left):    32 x 10 x 64
                       1 x  1 x 64
Step 3 (Stretch 1s):  32 x 10 x 64 <-- Resulting Shape!
```

**Mental Model:** PyTorch sees the `1`s and says, "Ah, I need to copy this bias vector across all 32 batches and all 10 time steps."

---

### Rule 2: The Matrix Rule (Linear Algebra Ops)

**Applies to:** `@`, `torch.matmul`, `nn.Linear`, `bmm`

This is the rule that confuses people the most in Transformers.

**The Rule:** PyTorch ONLY looks at the **last two dimensions** to do matrix multiplication. Everything to the left is treated as a "Batch" dimension, and PyTorch just runs a hidden `for` loop over them.

**The Formula:** `(..., M, K) @ (..., K, N) -> (..., M, N)`

#### Example: Self-Attention (Q @ K.transpose)

- **Query shape:** `(Batch=32, Heads=8, Time=10, Dim=64)`
- **Key shape:** `(Batch=32, Heads=8, Dim=64, Time=10)`

PyTorch ignores the `32` and the `8`. It just sees a stack of matrices. It says: "For every batch, and for every head, multiply a `(10 x 64)` matrix by a `(64 x 10)` matrix."

**Result:** `(32, 8, 10, 10)` -> Your attention scores!

---

### Rule 3: The Pointer Rule (Shape Ops)

**Applies to:** `transpose`, `view`, `reshape`, `unsqueeze`, `squeeze`

These operations **do not do math**. Tensors are actually stored in your RAM/GPU as one massive flat 1D array of numbers. The "shape" is just a metadata object that tells PyTorch how to jump through that 1D array.

- **`transpose(1, 2)`**: Doesn't move data. It just swaps the metadata pointers so PyTorch reads the memory in a different order.
- **`view(B, T, C)`**: Tells PyTorch to slice the 1D array into a 3D grid.

**The Danger:** If you use `.view()` to rearrange dimensions when you **should** have used `.transpose()`, PyTorch won't crash. It will just confidently read the memory blocks in the wrong order, mixing up your Batch data with your Time data.

---

### The CS Solution: Stop Guessing, Start Asserting

As a CS student, relying on implicit rules (like Broadcasting) is terrifying. If you want to code faster and stop worrying about "Did PyTorch batch this correctly?", you should adopt these two practices.

#### 1. Use `unsqueeze` to be explicit

Don't let PyTorch guess where to put the `1`s during broadcasting. Tell it exactly where they go.

```python
# Instead of this (hoping PyTorch aligns it right):
x = x + bias

# Do this (Explicitly declaring the shape):
# x is (B, T, C), bias is (C,)
x = x + bias.unsqueeze(0).unsqueeze(0) # bias is now (1, 1, C)
```

#### 2. The Ultimate Cheat Code: `einops`

`einops` forces you to write the `for` loop logic as a string.

```python
from einops import rearrange, einsum

# 1. Reshaping (Splitting Channels into Heads)
# Read it as: "Take C, and split it into (h * d). Then move 'h' before 't'"
Q = rearrange(x, 'b t (h d) -> b h t d', h=8)

# 2. Batched Matrix Multiplication (Q @ K.T)
# Read it as: "For every b and h, multiply a (t x d) matrix by a (d x t) matrix"
# It perfectly handles the batching without you needing to remember Rule 2.
scores = einsum(Q, K, 'b h t d, b h d t_key -> b h t t_key')
```

---

### Pro-CS Summary Table

| Category       | Typical Logic                | Mental Shortcut           |
| :------------- | :--------------------------- | :------------------------ |
| **Math**       | Right-align & Stretch `1`s.  | "Element-wise sync."      |
| **Matrices**   | $2D$ math + $ND$ Batch Loop. | "Stack of dot products."  |
| **Reductions** | Delete the `dim` index.      | "Squashing a coordinate." |
