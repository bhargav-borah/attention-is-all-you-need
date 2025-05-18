# Transformer from Scratch

A PyTorch implementation of the Transformer architecture from the paper "Attention Is All You Need" (Vaswani et al., 2017).

## Overview

This repository contains a complete implementation of the Transformer model for sequence-to-sequence tasks, specifically machine translation. The implementation follows the architecture described in the original paper while providing a clean, modular, and well-documented codebase.

Key features:
- Full implementation of the Transformer architecture
- Multi-head self-attention mechanism
- Positional encoding using sinusoidal functions
- Support for English-to-German translation
- Training, evaluation, and inference capabilities
- BLEU score calculation for model evaluation

## Repository Structure

```
transformer-from-scratch/
│
├── config/               # Configuration parameters
├── data/                 # Dataset and tokenizer implementations
├── model/                # Transformer model components
├── train/                # Training and evaluation utilities
├── utils/                # Helper functions
├── scripts/              # Training and evaluation scripts
└── notebooks/            # Jupyter notebooks for demos
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bhargav-borah/attention-is-all-you-need.git
cd attention-is-all-you-need
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model on the WMT14 English-to-German dataset:

```bash
python scripts/train.py
```

You can modify training parameters in the `config/default.py` file.

### Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate.py --model_path path/to/saved/model
```

### Translation

```python
from model.transformer import Transformer
from data.tokenizer import Tokenizer

# Load model and tokenizer
tokenizer = Tokenizer(max_len=50)
model = Transformer(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load('path/to/saved/model'))

# Translate a sentence
src_text = "Hello, how are you?"
translated = model.translate(src_text, tokenizer)
print(translated)
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017

## License

MIT
