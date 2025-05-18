import torch
import argparse
from data.dataset import load_data, get_dataloaders
from model.transformer import Transformer
from train.training import evaluate_model
from config.default import CFG

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    args = parser.parse_args()

    # Load data
    train_df, val_df, test_df = load_data()
    
    # Get dataloaders and tokenizer
    train_dl, val_dl, test_dl, tokenizer = get_dataloaders(train_df, val_df, test_df)
    
    # Initialize model
    model = Transformer(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(args.model_path))
    
    # Evaluate model
    test_loss, test_perplexity, test_bleu = evaluate_model(model, test_dl, tokenizer)

if __name__ == "__main__":
    main()