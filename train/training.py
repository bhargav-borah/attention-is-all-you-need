import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import calculate_bleu

def train_model(model, train_dl, val_dl, tokenizer, epochs=10, lr=0.0001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens[tokenizer.pad_token])

    train_losses, val_losses = [], []
    train_perplexities, val_perplexities = [], []
    train_bleus, val_bleus = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, total_bleu, total_tokens = 0, 0, 0
        for batch in We’re sorry, but the response was cut off. Based on the context and the user’s request, I’ll complete the artifact for `train/training.py` and provide the remaining files to ensure a well-organized GitHub repository that matches the desired structure. The user’s code will remain unchanged, split into modular files, with an MIT license and a README citing the "Attention Is All You Need" paper. I’ll also correct the typo in the original code (`spyeical_tokens` to `special_tokens`) in the `evaluate_model` function to ensure functionality, as this appears to be a clear error.

### Completing the `train/training.py` Artifact
The `train/training.py` file should contain the `train_model` and `evaluate_model` functions, with dependencies imported correctly. Below is the completed artifact, continuing from where it was cut off.

<xaiArtifact artifact_id="cc0b1401-fcb1-43f6-9af2-ed34a85d2014" artifact_version_id="c53dbba1-2e8c-43ea-b77f-dbb57ae8775f" title="train/training.py" contentType="text/python">
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import calculate_bleu

def train_model(model, train_dl, val_dl, tokenizer, epochs=10, lr=0.0001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens[tokenizer.pad_token])

    train_losses, val_losses = [], []
    train_perplexities, val_perplexities = [], []
    train_bleus, val_bleus = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, total_bleu, total_tokens = 0, 0, 0
        for batch in train_dl:
            optimizer.zero_grad()
            src_ids = batch['source_ids'].to(device)
            dec_input_ids = batch['decoder_input_ids'].to(device)
            tgt_ids = batch['target_ids'].to(device)

            src_padding_mask = (src_ids == tokenizer.special_tokens[tokenizer.pad_token]).to(device)
            tgt_padding_mask = (dec_input_ids == tokenizer.special_tokens[tokenizer.pad_token]).to(device)

            outputs = model(src_ids, dec_input_ids, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
            loss = criterion(outputs.view(-1, model.vocab_size), tgt_ids.view(-1))

            loss.backward()
            optimizer.step()

            non_pad_mask = (tgt_ids != tokenizer.special_tokens[tokenizer.pad_token]).float()
            num_tokens = non_pad_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            pred_ids = torch.argmax(outputs, dim=-1)
            bleu = calculate_bleu(pred_ids.cpu(), tgt_ids.cpu(), tokenizer)
            total_bleu += bleu * num_tokens

        avg_train_loss = total_loss / total_tokens
        avg_train_perplexity = math.exp(avg_train_loss)
        avg_train_bleu = total_bleu / total_tokens

        model.eval()
        total_val_loss, total_val_bleu, total_val_tokens = 0, 0, 0
        with torch.no_grad():
            for batch in val_dl:
                src_ids = batch['source_ids'].to(device)
                dec_input_ids = batch['decoder_input_ids'].to(device)
                tgt_ids = batch['target_ids'].to(device)

                src_padding_mask = (src_ids == tokenizer.special_tokens[tokenizer.pad_token]).to(device)
                tgt_padding_mask = (dec_input_ids == tokenizer.special_tokens[tokenizer.pad_token]).to(device)

                outputs = model(src_ids, dec_input_ids, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
                loss = criterion(outputs.view(-1, model.vocab_size), tgt_ids.view(-1))

                non_pad_mask = (tgt_ids != tokenizer.special_tokens[tokenizer.pad_token]).float()
                num_tokens = non_pad_mask.sum().item()
                total_val_loss += loss.item() * num_tokens
                total_val_tokens += num_tokens

                pred_ids = torch.argmax(outputs, dim=-1)
                bleu = calculate_bleu(pred_ids.cpu(), tgt_ids.cpu(), tokenizer)
                total_val_bleu += bleu * num_tokens

        avg_val_loss = total_val_loss / total_val_tokens
        avg_val_perplexity = math.exp(avg_val_loss)
        avg_val_bleu = total_val_bleu / total_val_tokens

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_perplexities.append(avg_train_perplexity)
        val_perplexities.append(avg_val_perplexity)
        train_bleus.append(avg_train_bleu)
        val_bleus.append(avg_val_bleu)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Perplexity: {avg_train_perplexity:.4f}, BLEU: {avg_train_bleu:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Perplexity: {avg_val_perplexity:.4f}, BLEU: {avg_val_bleu:.4f}')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(train_perplexities, label='Train Perplexity')
    plt.plot(val_perplexities, label='Val Perplexity')
    plt.title('Perplexity')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(train_bleus, label='Train BLEU')
    plt.plot(val_bleus, label='Val BLEU')
    plt.title('BLEU Score')
    plt.legend()
    plt.savefig('training_metrics.png')

    return train_losses, val_losses, train_perplexities, val_perplexities, train_bleus, val_bleus

def evaluate_model(model, test_dl, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens[tokenizer.pad_token])
    model.eval()
    total_loss, total_bleu, total_tokens = 0, 0, 0

    with torch.no_grad():
        for batch in test_dl:
            src_ids = batch['source_ids'].to(device)
            dec_input_ids = batch['decoder_input_ids'].to(device)
            tgt_ids = batch['target_ids'].to(device)

            src_padding_mask = (src_ids == tokenizer.special_tokens[tokenizer.pad_token]).to(device)
            tgt_padding_mask = (dec_input_ids == tokenizer.special_tokens[tokenizer.pad_token]).to(device)

            outputs = model(src_ids, dec_input_ids, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
            loss = criterion(outputs.view(-1, model.vocab_size), tgt_ids.view(-1))

            non_pad_mask = (tgt_ids != tokenizer.special_tokens[tokenizer.pad_token]).float()
            num_tokens = non_pad_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            pred_ids = torch.argmax(outputs, dim=-1)
            bleu = calculate_bleu(pred_ids.cpu(), tgt_ids.cpu(), tokenizer)
            total_bleu += bleu * num_tokens

    avg_loss = total_loss / total_tokens
    avg_perplexity = math.exp(avg_loss)
    avg_bleu = total_bleu / total_tokens

    print(f'Test Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}, BLEU: {avg_bleu:.4f}')
    return avg_loss, avg_perplexity, avg_bleu