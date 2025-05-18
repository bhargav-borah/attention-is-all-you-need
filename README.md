# Transformer from Scratch

A PyTorch implementation of the Transformer architecture from the paper "Attention Is All You Need" (Vaswani et al., 2017).

## Overview

This repository contains a complete implementation of the Transformer model for sequence-to-sequence tasks, specifically English-to-German translation. The implementation follows the architecture described in the original paper while providing a clean, modular, and well-documented codebase.

**Key features:**

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
│   └── default.py
├── data/                 # Dataset and tokenizer implementations
│   ├── dataset.py
│   └── tokenizer.py
├── model/                # Transformer model components
│   └── transformer.py
├── train/                # Training and evaluation utilities
│   └── training.py
├── utils/                # Helper functions
│   └── metrics.py
├── scripts/              # Training and evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── notebooks/            # Jupyter notebooks for demos
│   └── implementation-from-scratch.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/bhargav-borah/attention-is-all-you-need.git
cd attention-is-all-you-need
```

Install dependencies:

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
import torch

# Load model and tokenizer
tokenizer = Tokenizer(max_len=50)
model = Transformer(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load('path/to/saved/model'))

# Translate a sentence
src_text = "Hello, how are you?"
# Note: Translation method not provided in original code; placeholder for user implementation
# translated = model.translate(src_text, tokenizer)
# print(translated)
```

## References

- Attention Is All You Need - Vaswani, A., Shazeer, N., Parmar, N., Uszoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Advances in Neural Information Processing Systems (NeurIPS)*. arXiv:1706.03762

## License

MIT