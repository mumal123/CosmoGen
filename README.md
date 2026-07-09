# CosmoGen

A character-level language model that generates cosmetic brand names using both a **Bigram Probability Model** and a **Single-Layer Neural Network** built from scratch in PyTorch.

---

## Dataset

The original dataset was obtained from Gigasheet:

https://app.gigasheet.com/spreadsheet/free-list-of-cosmetics-businesses-csv/a718a8c2_838d_4d7d_b28e_472ec6312635?referrerId=https%3A%2F%2Fwww.gigasheet.com%2Fsample-data%2Ffree-list-of-cosmetics-businessescsv&_gl=1*1luvqnp*_gcl_au*NTk1ODgxNzk0LjE3ODMzOTY3MDI.

### Data Preprocessing

Before training, I cleaned the dataset by:

- Extracting only cosmetic brand names.
- Removing duplicate and invalid entries.
- Removing emojis and unsupported special characters.
- Keeping only alphabetic characters, spaces, apostrophes (`'`), hyphens (`-`).
- Building a custom vocabulary from the cleaned dataset.

---

## Project Pipeline

### 1. Bigram Language Model

Implemented a simple statistical language model by:

- Counting character-to-character transitions.
- Creating a bigram count matrix.
- Applying **Laplace (Add-One) Smoothing**.
- Converting counts into probability distributions.
- Sampling names using `torch.multinomial()`.
- Using `torch.Generator` for reproducible results.

---

### 2. Neural Network Language Model

Built a single-layer neural network from scratch without using `torch.nn`.

The model includes:

- One-Hot Encoding
- Matrix Multiplication
- Softmax
- Negative Log Likelihood Loss
- Gradient Descent
- Backpropagation
- L2 Regularization
- Random sampling using `torch.multinomial()`

The neural network learns nearly the same probability distribution as the statistical bigram model.

---

## Concepts Learned

During this project I learned about:

- Character-level Language Models
- Bigram Probability Models
- One-Hot Encoding
- Softmax
- Likelihood & Log-Likelihood
- Negative Log Likelihood (NLL)
- Laplace Smoothing
- L2 Regularization
- Gradient Descent
- Backpropagation
- Broadcasting in PyTorch
- Probability Sampling using `torch.multinomial`
- Reproducibility using `torch.Generator`

---

## Tech Stack

- Python
- PyTorch
- Matplotlib
- Jupyter Notebook

---

## Sample Generated Names

```
aeraliauizjquted.
batroudralityzcsg inmphtud shaag.
pxos so pais.
ltarvishorelcaty.
mindrt.
fetr m llsmet tin.
io.
```

> **Note:** Some generated names may seem weird—and that's completely expected! This project uses a simple character-level bigram model and a single-layer neural network, which only learn the probability of the next character. More advanced models like MLPs, RNNs, or Transformers can generate much more realistic brand names by capturing longer-range patterns.

---

## Inspiration

This project is inspired by Andrej Karpathy's excellent **makemore** series, where neural networks are built from first principles to generate names. Instead of using the original baby names dataset, I experimented with a custom cosmetics brand dataset and performed my own preprocessing before training the models.

