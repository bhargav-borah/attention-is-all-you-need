import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from .metrics import calculate_bleu


class Trainer:
    """
    Trainer class for training and evaluating Transformer models.
    
    Args:
        model (nn.Module): The Transformer model
        train_dataloader (DataLoader): DataLoader for training data
        val_dataloader (DataLoader): DataLoader for validation data
        tokenizer (Tokenizer): Tokenizer instance
        learning_rate (float, optional): Learning rate. Defaults to 0.0001.
        device (str, optional): Device to use. Defaults to 'cuda' if available.
    """
    def __init__(self, model, train_dataloader, val_dataloader, tokenizer, 
                 learning_rate=0.0001, device=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize loss function (ignore padding tokens)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.special_tokens[self.tokenizer.pad_token]
        )
        
        # Initialize tracking variables
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.train_bleus = []
        self.val_bleus = []
        
    def train(self, epochs=10, save_dir='checkpoints', save_freq=1):
        """
        Train the model for the given number of epochs.
        
        Args:
            epochs (int, optional): Number of training epochs. Defaults to 10.
            save_dir (str, optional): Directory to save model checkpoints. Defaults to 'checkpoints'.
            save_freq (int, optional): Frequency (in epochs) to save model checkpoints. Defaults to 1.
            
        Returns:
            dict: Dictionary containing training history
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss, total_bleu, total_tokens = 0, 0, 0
            
            # Wrap dataloader with tqdm for progress tracking
            train_iter = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for batch in train_iter:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Move batch to device
                src_ids = batch['source_ids'].to(self.device)
                dec_input_ids = batch['decoder_input_ids'].to(self.device)
                tgt_ids = batch['target_ids'].to(self.device)
                
                # Create padding masks
                src_padding_mask = (src_ids == self.tokenizer.special_tokens[self.tokenizer.pad_token]).to(self.device)
                tgt_padding_mask = (dec_input_ids == self.tokenizer.special_tokens[self.tokenizer.pad_token]).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    src_ids, 
                    dec_input_ids, 
                    src_padding_mask=src_padding_mask, 
                    tgt_padding_mask=tgt_padding_mask
                )
                
                # Calculate loss
                loss = self.criterion(outputs.view(-1, self.model.vocab_size), tgt_ids.view(-1))
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Calculate metrics
                non_pad_mask = (tgt_ids != self.tokenizer.special_tokens[self.tokenizer.pad_token]).float()
                num_tokens = non_pad_mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
                # Calculate BLEU score
                pred_ids = torch.argmax(outputs, dim=-1)
                bleu = calculate_bleu(pred_ids.cpu(), tgt_ids.cpu(), self.tokenizer)
                total_bleu += bleu * num_tokens
                
                # Update progress bar with current loss
                train_iter.set_postfix({'loss': loss.item()})
            
            # Calculate average metrics
            avg_train_loss = total_loss / total_tokens
            avg_train_perplexity = math.exp(avg_train_loss)
            avg_train_bleu = total_bleu / total_tokens
            
            # Validation phase
            avg_val_loss, avg_val_perplexity, avg_val_bleu = self.evaluate()
            
            # Store metrics
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_perplexities.append(avg_train_perplexity)
            self.val_perplexities.append(avg_val_perplexity)
            self.train_bleus.append(avg_train_bleu)
            self.val_bleus.append(avg_val_bleu)
            
            # Print metrics
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Perplexity: {avg_train_perplexity:.4f}, BLEU: {avg_train_bleu:.4f}')
            print(f'  Val Loss: {avg_val_loss:.4f}, Perplexity: {avg_val_perplexity:.4f}, BLEU: {avg_val_bleu:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(save_dir, f'transformer_epoch_{epoch+1}.pt')
                self.save_checkpoint(checkpoint_path)
                print(f'  Model saved to {checkpoint_path}')
        
        # Save final model
        final_model_path = os.path.join(save_dir, 'transformer_final.pt')
        self.save_checkpoint(final_model_path)
        print(f'Final model saved to {final_model_path}')
        
        # Plot training history
        self.plot_training_history(save_dir)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
            'train_bleus': self.train_bleus,
            'val_bleus': self.val_bleus
        }
    
    def evaluate(self, dataloader=None):
        """
        Evaluate the model on validation data.
        
        Args:
            dataloader (DataLoader, optional): DataLoader to use for evaluation.
                                             Defaults to None (use validation dataloader).
                                             
        Returns:
            tuple: (avg_loss, avg_perplexity, avg_bleu)
        """
        if dataloader is None:
            dataloader = self.val_dataloader
            
        self.model.eval()
        total_loss, total_bleu, total_tokens = 0, 0, 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                src_ids = batch['source_ids'].to(self.device)
                dec_input_ids = batch['decoder_input_ids'].to(self.device)
                tgt_ids = batch['target_ids'].to(self.device)
                
                # Create padding masks
                src_padding_mask = (src_ids == self.tokenizer.special_tokens[self.tokenizer.pad_token]).to(self.device)
                tgt_padding_mask = (dec_input_ids == self.tokenizer.special_tokens[self.tokenizer.pad_token]).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    src_ids, 
                    dec_input_ids, 
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask
                )
                
                # Calculate loss
                loss = self.criterion(outputs.view(-1, self.model.vocab_size), tgt_ids.view(-1))
                
                # Calculate metrics
                non_pad_mask = (tgt_ids != self.tokenizer.special_tokens[self.tokenizer.pad_token]).float()
                num_tokens = non_pad_mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
                # Calculate BLEU score
                pred_ids = torch.argmax(outputs, dim=-1)
                bleu = calculate_bleu(pred_ids.cpu(), tgt_ids.cpu(), self.tokenizer)
                total_bleu += bleu * num_tokens
        
        # Calculate average metrics
        avg_loss = total_loss / total_tokens
        avg_perplexity = math.exp(avg_loss)
        avg_bleu = total_bleu / total_tokens
        
        return avg_loss, avg_perplexity, avg_bleu
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        
        Args:
            path (str): Path to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
            'train_bleus': self.train_bleus,
            'val_bleus': self.val_bleus
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint.
        
        Args:
            path (str): Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history if available
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_perplexities = checkpoint.get('train_perplexities', [])
        self.val_perplexities = checkpoint.get('val_perplexities', [])
        self.train_bleus = checkpoint.get('train_bleus', [])
        self.val_bleus = checkpoint.get('val_bleus', [])
    
    def plot_training_history(self, save_dir='.'):
        """
        Plot training history metrics.
        
        Args:
            save_dir (str, optional): Directory to save the plot. Defaults to '.'.
        """
        plt.figure(figsize=(16, 6))
        
        # Plot Loss
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot Perplexity
        plt.subplot(1, 3, 2)
        plt.plot(self.train_perplexities, label='Train Perplexity')
        plt.plot(self.val_perplexities, label='Val Perplexity')
        plt.title('Perplexity')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot BLEU Score
        plt.subplot(1, 3, 3)
        plt.plot(self.train_bleus, label='Train BLEU')
        plt.plot(self.val_bleus, label='Val BLEU')
        plt.title('BLEU Score')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
        plt.close()