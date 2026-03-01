"""
Medical Chatbot Fine-Tuning and Evaluation Script
===================================================
Fine-tunes a medical chatbot on the merged QA dataset and evaluates performance.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
from collections import Counter
from datetime import datetime
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Model architecture - optimized for faster training
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "num_heads": 4,
    "dropout": 0.2,
    "max_seq_len": 64,
    
    # Training - reduced for faster iteration
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "gradient_clip": 1.0,
    
    # Data
    "min_word_freq": 3,
    "train_split": 0.90,
    "val_split": 0.05,
    "test_split": 0.05,
    
    # Paths
    "data_path": r"D:\project 2\data\chatbot_training_combined\chatbot_training_combined.json",
    "output_dir": r"D:\project 2\data\chatbot_training_combined\training_results",
    "checkpoint_path": r"D:\project 2\checkpoints\medical_chatbot_finetuned.pth",
}

# =============================================================================
# VOCABULARY
# =============================================================================

class Vocabulary:
    """Vocabulary for tokenization with special tokens."""
    
    def __init__(self):
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
        self.word2idx = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.idx2word = {i: tok for tok, i in self.word2idx.items()}
        self.word_counts = Counter()
        self.vocab_size = len(self.special_tokens)
    
    def add_sentence(self, sentence):
        """Add words from sentence to vocabulary."""
        for word in self._tokenize(sentence):
            self.word_counts[word] += 1
    
    def _tokenize(self, text):
        """Simple tokenization."""
        text = text.lower().strip()
        # Replace punctuation with spaces
        for char in '.,!?;:()[]{}"\'':
            text = text.replace(char, f' {char} ')
        return text.split()
    
    def build_vocab(self, min_freq=1):
        """Build vocabulary from word counts."""
        for word, count in self.word_counts.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text, max_len=128, add_special=True):
        """Encode text to indices."""
        tokens = self._tokenize(text)
        indices = [self.word2idx.get(w, 1) for w in tokens]  # 1 = <UNK>
        
        if add_special:
            indices = [2] + indices + [3]  # <BOS> and <EOS>
        
        # Pad or truncate
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len-1] + [3]  # Keep EOS
        
        return indices
    
    def decode(self, indices):
        """Decode indices to text."""
        words = []
        for i in indices:
            if i == 0:  # PAD
                continue
            if i == 3:  # EOS
                break
            if i == 2:  # BOS
                continue
            words.append(self.idx2word.get(i, "<UNK>"))
        return " ".join(words)


# =============================================================================
# DATASET
# =============================================================================

class MedicalQADataset(Dataset):
    """Dataset for medical QA pairs."""
    
    def __init__(self, qa_pairs, vocab, max_len=128):
        self.qa_pairs = qa_pairs
        self.vocab = vocab
        self.max_len = max_len
        
        # Pre-encode all data
        self.encoded_data = []
        for qa in qa_pairs:
            question = qa['question']
            answer = qa['answer']
            category = qa.get('category', 'general')
            
            # Encode question and answer
            q_encoded = vocab.encode(question, max_len, add_special=True)
            a_encoded = vocab.encode(answer, max_len, add_special=True)
            
            # Combined sequence for training
            combined = q_encoded[:max_len//2] + a_encoded[:max_len//2]
            if len(combined) < max_len:
                combined += [0] * (max_len - len(combined))
            
            self.encoded_data.append({
                'question': q_encoded,
                'answer': a_encoded,
                'combined': combined,
                'category': category,
                'question_text': question,
                'answer_text': answer
            })
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        item = self.encoded_data[idx]
        return {
            'question': torch.LongTensor(item['question']),
            'answer': torch.LongTensor(item['answer']),
            'combined': torch.LongTensor(item['combined']),
            'category': item['category']
        }


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class MedicalChatbotModel(nn.Module):
    """
    Transformer-based medical chatbot model.
    
    Architecture:
    - Token embeddings with positional encoding
    - Stack of transformer blocks
    - Language modeling head for generation
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim * 2, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Shift logits and targets for language modeling
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_targets.view(-1),
                ignore_index=0  # Ignore padding
            )
        
        return logits, loss
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=50):
        """Generate response tokens."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                logits, _ = self.forward(input_ids)
                
                # Get last token logits
                logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop on EOS
                if next_token.item() == 3:  # <EOS>
                    break
                
                # Stop if too long
                if input_ids.size(1) >= self.max_seq_len:
                    break
        
        return input_ids


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def load_qa_dataset(data_path):
    """Load the merged QA dataset."""
    print("=" * 70)
    print("LOADING MEDICAL QA DATASET")
    print("=" * 70)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = data['qa_pairs']
    print(f"\nTotal QA pairs loaded: {len(qa_pairs)}")
    
    # Category distribution
    categories = Counter(qa.get('category', 'unknown') for qa in qa_pairs)
    print(f"\nCategory distribution:")
    for cat, count in categories.most_common():
        print(f"   {cat}: {count}")
    
    return qa_pairs, categories


def prepare_data_splits(qa_pairs, vocab, config):
    """Prepare train/val/test splits."""
    print("\n" + "=" * 70)
    print("PREPARING DATA SPLITS")
    print("=" * 70)
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    for qa in qa_pairs:
        vocab.add_sentence(qa['question'])
        vocab.add_sentence(qa['answer'])
    
    vocab.build_vocab(min_freq=config['min_word_freq'])
    print(f"Vocabulary size: {vocab.vocab_size}")
    
    # Shuffle data
    random.shuffle(qa_pairs)
    
    # Split
    n = len(qa_pairs)
    train_end = int(n * config['train_split'])
    val_end = train_end + int(n * config['val_split'])
    
    train_data = qa_pairs[:train_end]
    val_data = qa_pairs[train_end:val_end]
    test_data = qa_pairs[val_end:]
    
    print(f"\nDataset splits:")
    print(f"   Training: {len(train_data)}")
    print(f"   Validation: {len(val_data)}")
    print(f"   Test: {len(test_data)}")
    
    # Create datasets
    train_dataset = MedicalQADataset(train_data, vocab, config['max_seq_len'])
    val_dataset = MedicalQADataset(val_data, vocab, config['max_seq_len'])
    test_dataset = MedicalQADataset(test_data, vocab, config['max_seq_len'])
    
    return train_dataset, val_dataset, test_dataset


def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        input_ids = batch['combined'].to(device)
        
        optimizer.zero_grad()
        
        logits, loss = model(input_ids, targets=input_ids)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['combined'].to(device)
            
            logits, loss = model(input_ids, targets=input_ids)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def calculate_perplexity(model, dataloader, device):
    """Calculate perplexity on dataset."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['combined'].to(device)
            
            logits, loss = model(input_ids, targets=input_ids)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss


def evaluate_by_category(model, test_dataset, vocab, device):
    """Evaluate accuracy by category."""
    model.eval()
    
    category_metrics = {}
    
    # Group by category
    categories = {}
    for i in range(len(test_dataset)):
        item = test_dataset.encoded_data[i]
        cat = item['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(i)
    
    # Evaluate each category
    for cat, indices in categories.items():
        total = len(indices)
        correct_tokens = 0
        total_tokens = 0
        
        for idx in indices:
            item = test_dataset.encoded_data[idx]
            input_ids = torch.LongTensor(item['combined']).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, _ = model(input_ids)
            
            predicted = torch.argmax(logits, dim=-1)
            
            # Count correct tokens (excluding padding)
            target = torch.LongTensor(item['combined']).to(device)
            mask = target != 0
            
            correct_tokens += (predicted[0][mask] == target[mask]).sum().item()
            total_tokens += mask.sum().item()
        
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        category_metrics[cat] = {
            'total_samples': total,
            'token_accuracy': accuracy
        }
    
    return category_metrics


def generate_samples(model, vocab, test_dataset, device, num_samples=5):
    """Generate sample responses."""
    model.eval()
    samples = []
    
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    for idx in indices:
        item = test_dataset.encoded_data[idx]
        question = item['question_text']
        reference = item['answer_text']
        
        # Encode question
        input_ids = torch.LongTensor(item['question']).unsqueeze(0).to(device)
        
        # Generate
        output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=40)
        
        generated = vocab.decode(output_ids[0].tolist())
        
        samples.append({
            'question': question,
            'reference_answer': reference[:200] + '...' if len(reference) > 200 else reference,
            'generated_answer': generated
        })
    
    return samples


def save_training_report(config, vocab, train_stats, eval_results, category_metrics, samples, output_dir):
    """Save comprehensive training report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'configuration': config,
        'vocabulary': {
            'size': vocab.vocab_size,
            'special_tokens': vocab.special_tokens
        },
        'training_statistics': train_stats,
        'evaluation_results': eval_results,
        'category_metrics': category_metrics,
        'sample_outputs': samples,
        'status': 'completed'
    }
    
    report_path = os.path.join(output_dir, 'training_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nTraining report saved to: {report_path}")
    
    # Also save human-readable summary
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MEDICAL CHATBOT TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Timestamp: {report['timestamp']}\n\n")
        
        f.write("DATASET STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training samples: {train_stats['train_samples']}\n")
        f.write(f"Validation samples: {train_stats['val_samples']}\n")
        f.write(f"Test samples: {train_stats['test_samples']}\n")
        f.write(f"Vocabulary size: {vocab.vocab_size}\n\n")
        
        f.write("TRAINING RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Epochs trained: {train_stats['epochs_trained']}\n")
        f.write(f"Best validation loss: {train_stats['best_val_loss']:.4f}\n")
        f.write(f"Final training loss: {train_stats['final_train_loss']:.4f}\n\n")
        
        f.write("EVALUATION RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test perplexity: {eval_results['test_perplexity']:.2f}\n")
        f.write(f"Test loss: {eval_results['test_loss']:.4f}\n\n")
        
        f.write("ACCURACY BY CATEGORY\n")
        f.write("-" * 40 + "\n")
        for cat, metrics in sorted(category_metrics.items(), key=lambda x: x[1]['token_accuracy'], reverse=True):
            f.write(f"  {cat}: {metrics['token_accuracy']*100:.2f}% ({metrics['total_samples']} samples)\n")
        
        f.write("\n\nSAMPLE OUTPUTS\n")
        f.write("-" * 40 + "\n")
        for i, sample in enumerate(samples, 1):
            f.write(f"\nSample {i}:\n")
            f.write(f"  Q: {sample['question']}\n")
            f.write(f"  Ref: {sample['reference_answer']}\n")
            f.write(f"  Gen: {sample['generated_answer']}\n")
    
    print(f"Training summary saved to: {summary_path}")


def main():
    """Main training function."""
    print("\n" + "=" * 70)
    print("MEDICAL CHATBOT FINE-TUNING")
    print("=" * 70)
    
    # Set random seeds
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load data
    qa_pairs, categories = load_qa_dataset(CONFIG['data_path'])
    
    # Create vocabulary
    vocab = Vocabulary()
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_data_splits(qa_pairs, vocab, CONFIG)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    model = MedicalChatbotModel(
        vocab_size=vocab.vocab_size,
        embed_dim=CONFIG['embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        max_seq_len=CONFIG['max_seq_len'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel architecture:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Embedding dimension: {CONFIG['embed_dim']}")
    print(f"   Hidden dimension: {CONFIG['hidden_dim']}")
    print(f"   Transformer layers: {CONFIG['num_layers']}")
    print(f"   Attention heads: {CONFIG['num_heads']}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * CONFIG['epochs']
    
    def lr_lambda(step):
        if step < CONFIG['warmup_steps']:
            return step / CONFIG['warmup_steps']
        return 1.0 - step / total_steps
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, CONFIG['gradient_clip']
        )
        val_loss = evaluate(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(CONFIG['checkpoint_path']), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab.word2idx,
                'config': CONFIG,
                'val_loss': val_loss,
            }, CONFIG['checkpoint_path'])
    
    print("\nTraining complete!")
    
    # Evaluation
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    # Calculate perplexity
    test_perplexity, test_loss = calculate_perplexity(model, test_loader, device)
    print(f"\nTest Results:")
    print(f"   Perplexity: {test_perplexity:.2f}")
    print(f"   Loss: {test_loss:.4f}")
    
    # Category metrics
    print("\nEvaluating by category...")
    category_metrics = evaluate_by_category(model, test_dataset, vocab, device)
    
    print("\nAccuracy by Category:")
    for cat, metrics in sorted(category_metrics.items(), key=lambda x: x[1]['token_accuracy'], reverse=True)[:10]:
        print(f"   {cat}: {metrics['token_accuracy']*100:.2f}%")
    
    # Generate samples
    print("\nGenerating sample outputs...")
    samples = generate_samples(model, vocab, test_dataset, device, num_samples=5)
    
    print("\nSample Outputs:")
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}. Q: {sample['question'][:100]}...")
        print(f"   Generated: {sample['generated_answer'][:100]}...")
    
    # Save report
    train_stats = {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'epochs_trained': CONFIG['epochs'],
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    
    eval_results = {
        'test_perplexity': test_perplexity,
        'test_loss': test_loss,
    }
    
    save_training_report(
        CONFIG, vocab, train_stats, eval_results, category_metrics, samples, CONFIG['output_dir']
    )
    
    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {CONFIG['checkpoint_path']}")
    print(f"Report saved to: {CONFIG['output_dir']}")
    
    return model, vocab


if __name__ == "__main__":
    main()
