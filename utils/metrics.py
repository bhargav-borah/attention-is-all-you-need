from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

def calculate_bleu(pred_ids, target_ids, tokenizer):
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    for pred, tgt in zip(pred_ids, target_ids):
        pred_text = tokenizer.decode(pred.tolist())
        tgt_text = tokenizer.decode(tgt.tolist())
        pred_tokens = pred_text.split()
        tgt_tokens = tgt_text.split()
        if len(pred_tokens) == 0 or len(tgt_tokens) == 0:
            continue
        bleu = sentence_bleu([tgt_tokens], pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)
    return np.mean(bleu_scores) if bleu_scores else 0.0