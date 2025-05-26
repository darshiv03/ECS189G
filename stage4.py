import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
from tqdm import tqdm
import time
import string

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ImprovedTextDataset(Dataset):
    def __init__(self, file_path, seq_length=30, char_level=False, min_freq=1):
        self.seq_length = seq_length
        self.char_level = char_level
        self.min_freq = min_freq
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.raw_text = f.read()
        
        self.text = self.clean_text_conservative(self.raw_text)
        print(f"Original text length: {len(self.raw_text)} characters")
        print(f"Cleaned text length: {len(self.text)} characters")
        
        if char_level:
            self.prepare_char_level()
        else:
            self.prepare_word_level_improved()
        
        self.create_sequences()
    
    def clean_text_conservative(self, text):
        text = text.lower()
        text = re.sub(r'\n+', ' ', text)  
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def prepare_word_level_improved(self):
        words = self.text.split()
        print(f"Total words in text: {len(words)}")
        word_counts = Counter(words)
        print(f"Unique words: {len(word_counts)}")
        
        vocab_words = [word for word, count in word_counts.items() 
                      if count >= self.min_freq and len(word) > 1]
        
        self.words = ['<PAD>', '<UNK>', '<START>', '<END>'] + vocab_words
        self.vocab_size = len(self.words)
        
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.idx_to_word = {i: word for i, word in enumerate(self.words)}
        
        self.encoded = []
        unk_count = 0
        for word in words:
            if word in self.word_to_idx:
                self.encoded.append(self.word_to_idx[word])
            else:
                self.encoded.append(1)  
                unk_count += 1
        
        unk_percentage = (unk_count / len(words)) * 100
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"UNK tokens: {unk_count}/{len(words)} ({unk_percentage:.1f}%)")
        print(f"Sample words: {self.words[4:24]}")  
            
    def prepare_char_level(self):
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.encoded = [self.char_to_idx[ch] for ch in self.text]
        print(f"Character vocabulary size: {self.vocab_size}")
        print(f"Sample characters: {self.chars}")
    
    def create_sequences(self):
        self.sequences = []
        for i in range(len(self.encoded) - self.seq_length):
            seq_in = self.encoded[i:i + self.seq_length]
            seq_out = self.encoded[i + 1:i + self.seq_length + 1]
            self.sequences.append((seq_in, seq_out))
        
        print(f"Created {len(self.sequences)} sequences of length {self.seq_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_in, seq_out = self.sequences[idx]
        return torch.tensor(seq_in, dtype=torch.long), torch.tensor(seq_out, dtype=torch.long)
    
    def decode_sequence(self, sequence):
        if self.char_level:
            return ''.join([self.idx_to_char.get(idx, '?') for idx in sequence])
        else:
            return ' '.join([self.idx_to_word.get(idx, '<UNK>') for idx in sequence])

class ImprovedRNNGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, 
                 rnn_type='LSTM', dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:  
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0, nonlinearity='tanh')
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
      for name, param in self.named_parameters():
          if param.dim() >= 2 and 'weight' in name:
              nn.init.xavier_uniform_(param)
          elif 'bias' in name or param.dim() == 1:
              nn.init.zeros_(param)

        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        if self.rnn_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded, hidden)
            hidden_state = (hidden, cell)
        else:
            output, hidden = self.rnn(embedded, hidden)
            hidden_state = hidden
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        output = self.fc(output)
        
        return output, hidden_state
    
    def init_hidden(self, batch_size, device):
        if self.rnn_type == 'LSTM':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return (h0, c0)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

def generate_text_improved(model, dataset, start_text, length=200, temperature=0.8, top_k=50, top_p=0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        if dataset.char_level:
            current_seq = [dataset.char_to_idx.get(c, 0) for c in start_text.lower()]
        else:
            words = start_text.lower().split()
            current_seq = []
            for word in words:
                if word in dataset.word_to_idx:
                    current_seq.append(dataset.word_to_idx[word])
                else:
                    similar_found = False
                    for vocab_word in dataset.word_to_idx:
                        if vocab_word.startswith(word[:3]) and len(vocab_word) > 3:
                            current_seq.append(dataset.word_to_idx[vocab_word])
                            similar_found = True
                            break
                    if not similar_found:
                        current_seq.append(4)  
        
        if not current_seq:
            current_seq = [4, 5, 6]  
        
        generated = current_seq.copy()
        hidden = model.init_hidden(1, device)
        
        consecutive_unk = 0
        max_consecutive_unk = 3
        
        for _ in range(length):
            if len(current_seq) > dataset.seq_length:
                input_seq = current_seq[-dataset.seq_length:]
            else:
                input_seq = current_seq
            
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
            
            output, hidden = model(input_tensor, hidden)
            logits = output[0, -1, :] / temperature
            
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_logits
                logits = filtered_logits
            
            if consecutive_unk >= max_consecutive_unk:
                logits[1] = float('-inf')  
            
            probs = F.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == 1: 
                consecutive_unk += 1
            else:
                consecutive_unk = 0
            
            generated.append(next_token)
            current_seq.append(next_token)
            
            if not dataset.char_level and next_token == 3:  
                break
        
        return dataset.decode_sequence(generated)

def train_improved_generator(model, dataset, epochs=20, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            hidden = model.init_hidden(inputs.size(0), device)
            
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if (epoch + 1) % 5 == 0:
            sample_text = generate_text_improved(model, dataset, "the", length=100, temperature=0.7)
            print(f'Sample generation: {sample_text[:150]}...\n')
    
    return train_losses

def run_improved_experiment(data_path, starting_words=["the", "once", "in"], use_char_level=False):
    
    print(f"\n Loading dataset ({'character' if use_char_level else 'word'}-level)...")
    dataset = ImprovedTextDataset(data_path, seq_length=25, char_level=use_char_level, min_freq=1)
    
    if not use_char_level and hasattr(dataset, 'encoded'):
        unk_count = sum(1 for token in dataset.encoded if token == 1)
        unk_percentage = (unk_count / len(dataset.encoded)) * 100
        if unk_percentage > 25:
            print(f"\n  UNK percentage too high ({unk_percentage:.1f}%), switching to character-level...")
            dataset = ImprovedTextDataset(data_path, seq_length=50, char_level=True)
    
    model_configs = [
        {'rnn_type': 'RNN', 'hidden_dim': 512, 'num_layers': 2, 'embed_dim': 256},
        {'rnn_type': 'LSTM', 'hidden_dim': 512, 'num_layers': 2, 'embed_dim': 256},
        {'rnn_type': 'GRU', 'hidden_dim': 512, 'num_layers': 2, 'embed_dim': 256},
    ]
    
    results = {}
    
    for config in model_configs:
        print(f"\nTraining {config['rnn_type']} Model...")
        print("-" * 40)
        
        model = ImprovedRNNGenerator(
            vocab_size=dataset.vocab_size,
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            rnn_type=config['rnn_type'],
            dropout=0.2
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        start_time = time.time()
        train_losses = train_improved_generator(model, dataset, epochs=15, batch_size=32)
        training_time = time.time() - start_time
        
        perplexity = calculate_perplexity(model, dataset)
        
        generated_stories = {}
        
        for start_word in starting_words:
            print(f"\nGenerating story starting with '{start_word}'...")
            generated_text = generate_text_improved(model, dataset, start_word, 
                                                   length=150, temperature=0.7, top_k=50)
            generated_stories[start_word] = generated_text
            
            print(f"Generated story ({start_word}):")
            print("-" * 30)
            print(generated_text[:300] + "..." if len(generated_text) > 300 else generated_text)
            print()
        
        results[config['rnn_type']] = {
            'train_losses': train_losses,
            'training_time': training_time,
            'perplexity': perplexity,
            'generated_stories': generated_stories,
            'final_loss': train_losses[-1]
        }
        
        print(f"Final Training Loss: {train_losses[-1]:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Training Time: {training_time:.2f} seconds")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for model_type, result in results.items():
        plt.plot(result['train_losses'], label=f'{model_type}', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    model_types = list(results.keys())
    perplexities = [results[mt]['perplexity'] for mt in model_types]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = plt.bar(model_types, perplexities, color=colors)
    plt.ylabel('Perplexity (lower is better)')
    plt.title('Final Perplexity Comparison')
    for bar, perp in zip(bars, perplexities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{perp:.2f}', ha='center', va='bottom')
    
    plt.subplot(1, 3, 3)
    training_times = [results[mt]['training_time'] for mt in model_types]
    bars = plt.bar(model_types, training_times, color=colors)
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    for bar, time_val in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{time_val:.0f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\nFIXED Performance Metrics:")
    print("-" * 50)
    for model_type, result in results.items():
        print(f"\n{model_type} Model:")
        print(f"  Final Loss: {result['final_loss']:.4f}")
        print(f"  Perplexity: {result['perplexity']:.2f}")
        print(f"  Training Time: {result['training_time']:.1f}s")
    
    return results

def calculate_perplexity(model, dataset, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0), device)
            
            outputs, _ = model(inputs, hidden)
            loss = criterion(outputs.view(-1, model.vocab_size), targets.view(-1))
            
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

if __name__ == "__main__":
    data_path = "stage_4_data/stage_4_data/text_generation/data"
    
    try:
        print("Running FIXED text generation experiment...")
        results = run_improved_experiment(data_path, starting_words=["the", "once", "in"])
        
        print("\n Fixed experiment completed!")
        print("\n Key Improvements Made:")
        print("- Reduced minimum word frequency to include more vocabulary")
        print("- Added UNK token suppression during generation")
        print("- Improved text cleaning to preserve more content")
        print("- Enhanced sampling with top-k filtering")
        print("- Added word similarity matching for unknown words")
        print("- Automatic fallback to character-level if too many UNKs")
        
    except FileNotFoundError:
        print(f"Creating sample data for testing...")
        
        sample_stories = [
            "Once upon a time there was a brave knight who lived in a castle. The knight had a magical sword that could defeat any enemy. One day the knight went on a quest to save the kingdom from a terrible dragon.",
            "In a small village by the sea lived a young fisherman. Every morning he would sail out to catch fish for the market. The fisherman knew all the best spots where the biggest fish could be found.",
            "The old wizard sat in his tower studying ancient books of magic. He had spent many years learning powerful spells and potions. When the kingdom was in danger the wizard would use his magic to help.",
            "A clever merchant traveled from town to town selling his wares. He had beautiful silks from distant lands and precious jewels that sparkled in the sunlight. The merchant told amazing stories of his adventures.",
            "Deep in the forest lived a family of bears. The father bear was big and strong while the mother bear was gentle and kind. Their little cub loved to play and explore the woods around their home."
        ]
        
        full_text = " ".join(sample_stories * 20)  
        
        with open("sample_text_data.txt", "w") as f:
            f.write(full_text)
        
        print("Created enhanced sample_text_data.txt")
        results = run_improved_experiment("sample_text_data.txt", starting_words=["the", "once", "in"])
