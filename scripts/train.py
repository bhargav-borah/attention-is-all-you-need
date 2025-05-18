from data.dataset import load_data, get_dataloaders
from model.transformer import Transformer
from train.training import train_model
from config.default import CFG

def main():
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Get dataloaders and tokenizer
    train_dl, val_dl, test_dl, tokenizer = get_dataloaders(train_df, val_df, test_df)
    
    # Initialize model
    model = Transformer(vocab_size=tokenizer.vocab_size)
    
    # Train model
    train_losses, val_losses, train_perplexities, val_perplexities, train_bleus, val_bleus = train_model(
        model, train_dl, val_dl, tokenizer, epochs=3)

if __name__ == "__main__":
    main()