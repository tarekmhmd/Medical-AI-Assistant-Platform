"""
Fast Medical Chatbot Training and Evaluation
=============================================
Simplified training for quick results.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datetime import datetime
import random
import math

# Configuration - optimized for speed
CONFIG = {
    "embed_dim": 64,
    "hidden_dim": 128,
    "num_layers": 1,
    "max_seq_len": 32,
    "batch_size": 128,
    "epochs": 10,
    "learning_rate": 0.01,
    "min_word_freq": 5,
    "data_path": r"D:\project 2\data\chatbot_training_combined\chatbot_training_combined.json",
    "output_dir": r"D:\project 2\data\chatbot_training_combined\training_results",
    "checkpoint_path": r"D:\project 2\checkpoints\medical_chatbot_fast.pth",
}

class SimpleVocab:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_counts = Counter()
        self.vocab_size = 4
    
    def add_text(self, text):
        for word in text.lower().split():
            self.word_counts[word] += 1
    
    def build(self, min_freq=1):
        for word, count in self.word_counts.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text, max_len=32):
        tokens = [self.word2idx.get(w, 1) for w in text.lower().split()[:max_len-2]]
        tokens = [2] + tokens + [3]  # BOS, EOS
        tokens = tokens[:max_len] + [0] * max(0, max_len - len(tokens))
        return tokens[:max_len]
    
    def decode(self, ids):
        return " ".join(self.idx2word.get(i, "") for i in ids if i > 3)

class SimpleDataset(Dataset):
    def __init__(self, qa_pairs, vocab, max_len=32):
        self.data = []
        for qa in qa_pairs:
            q = vocab.encode(qa['question'], max_len)
            a = vocab.encode(qa['answer'], max_len)
            combined = (q + a)[:max_len] + [0] * max(0, max_len - len(q + a))
            self.data.append({
                'input': torch.LongTensor(combined[:max_len]),
                'question': qa['question'],
                'answer': qa['answer'][:200],
                'category': qa.get('category', 'general')
            })
    
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size
    
    def forward(self, x, targets=None):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        logits = self.fc(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=0
            )
        return logits, loss

def main():
    print("=" * 60)
    print("FAST MEDICAL CHATBOT TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading dataset...")
    with open(CONFIG['data_path'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = data['qa_pairs']
    print(f"   Loaded {len(qa_pairs)} QA pairs")
    
    # Build vocab
    print("\n[2/6] Building vocabulary...")
    vocab = SimpleVocab()
    for qa in qa_pairs[:30000]:  # Use subset for speed
        vocab.add_text(qa['question'])
        vocab.add_text(qa['answer'])
    vocab.build(min_freq=CONFIG['min_word_freq'])
    print(f"   Vocabulary size: {vocab.vocab_size}")
    
    # Split data
    print("\n[3/6] Splitting dataset...")
    random.shuffle(qa_pairs)
    n = len(qa_pairs)
    train_data = qa_pairs[:int(n*0.9)][:25000]  # Limit for speed
    val_data = qa_pairs[int(n*0.9):int(n*0.95)]
    test_data = qa_pairs[int(n*0.95):]
    
    train_ds = SimpleDataset(train_data, vocab, CONFIG['max_seq_len'])
    val_ds = SimpleDataset(val_data, vocab, CONFIG['max_seq_len'])
    test_ds = SimpleDataset(test_data, vocab, CONFIG['max_seq_len'])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'])
    
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Create model
    print("\n[4/6] Creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(vocab.vocab_size, CONFIG['embed_dim'], CONFIG['hidden_dim']).to(device)
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print("\n[5/6] Training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch['input'].to(device)
            optimizer.zero_grad()
            _, loss = model(x, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input'].to(device)
                _, loss = model(x, x)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(CONFIG['checkpoint_path']), exist_ok=True)
            torch.save(model.state_dict(), CONFIG['checkpoint_path'])
        
        print(f"   Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    # Evaluation
    print("\n[6/6] Evaluating...")
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch['input'].to(device)
            _, loss = model(x, x)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    perplexity = math.exp(test_loss)
    
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Perplexity: {perplexity:.2f}")
    
    # Category metrics
    category_correct = Counter()
    category_total = Counter()
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['input'].to(device)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=-1)
            
            for i, cat in enumerate(batch['category']):
                mask = x[i] != 0
                correct = (preds[i][mask] == x[i][mask]).sum().item()
                total = mask.sum().item()
                category_correct[cat] += correct
                category_total[cat] += total
    
    category_metrics = {}
    for cat in category_total:
        if category_total[cat] > 0:
            category_metrics[cat] = {
                'accuracy': category_correct[cat] / category_total[cat],
                'samples': category_total[cat]
            }
    
    # Generate samples
    samples = []
    model.eval()
    for i in range(min(5, len(test_ds))):
        item = test_ds[i]
        samples.append({
            'question': item['question'],
            'reference': item['answer'],
            'generated': vocab.decode(item['input'].tolist())
        })
    
    # Save report
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'configuration': CONFIG,
        'vocabulary_size': vocab.vocab_size,
        'dataset': {
            'train_samples': len(train_ds),
            'val_samples': len(val_ds),
            'test_samples': len(test_ds)
        },
        'training': {
            'epochs': CONFIG['epochs'],
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'train_losses': train_losses,
            'val_losses': val_losses
        },
        'evaluation': {
            'test_loss': test_loss,
            'test_perplexity': perplexity
        },
        'category_metrics': category_metrics,
        'sample_outputs': samples
    }
    
    report_path = os.path.join(CONFIG['output_dir'], 'training_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary_path = os.path.join(CONFIG['output_dir'], 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("MEDICAL CHATBOT TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {report['timestamp']}\n\n")
        f.write(f"Dataset: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}\n")
        f.write(f"Vocabulary: {vocab.vocab_size} tokens\n\n")
        f.write(f"Training Results:\n")
        f.write(f"  Epochs: {CONFIG['epochs']}\n")
        f.write(f"  Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"  Final Train Loss: {train_losses[-1]:.4f}\n\n")
        f.write(f"Evaluation Results:\n")
        f.write(f"  Test Loss: {test_loss:.4f}\n")
        f.write(f"  Test Perplexity: {perplexity:.2f}\n\n")
        f.write(f"Accuracy by Category:\n")
        for cat, m in sorted(category_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:10]:
            f.write(f"  {cat}: {m['accuracy']*100:.2f}%\n")
        f.write(f"\nSample Outputs:\n")
        for i, s in enumerate(samples, 1):
            f.write(f"\n{i}. Q: {s['question'][:80]}...\n")
            f.write(f"   Generated: {s['generated'][:80]}...\n")
    
    print(f"\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved: {CONFIG['checkpoint_path']}")
    print(f"Report saved: {report_path}")
    print(f"Summary saved: {summary_path}")
    
    return report

if __name__ == "__main__":
    main()
