"""
Train Full-Text OCR Model - Generate Complete Text from Images
Sequence-to-Sequence based model for prescription and lab report text recognition
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class OCRDataset(Dataset):
    """Load dataset with full text content"""
    
    def __init__(self, data_path, doc_type, split='train'):
        self.image_dir = f"{data_path}/{doc_type}/{split}"
        self.samples = []
        self.vocab = set()
        
        if os.path.exists(self.image_dir):
            for f in sorted(os.listdir(self.image_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(self.image_dir, f)
                    txt_name = f.rsplit('.', 1)[0] + '.txt'
                    txt_path = os.path.join(self.image_dir, txt_name)
                    
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
                                text = file.read().strip()
                            if text:
                                self.samples.append((img_path, text))
                                self.vocab.update(text)
                        except:
                            pass
        
        # Build vocabulary
        self.vocab = sorted(list(self.vocab))
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('L')
            image = image.resize((256, 64))
            image = torch.FloatTensor(np.array(image)) / 255.0
            
            # Convert text to indices and pad to fixed length
            text_indices = [self.char2idx.get(c, 0) for c in text[:256]]
            # Pad to 256 characters
            while len(text_indices) < 256:
                text_indices.append(0)
            text_indices = torch.tensor(text_indices[:256])
            
            return image, text_indices, text
        except:
            return None, None, None


class TextEncoder(nn.Module):
    """Encode image to text features"""
    
    def __init__(self):
        super(TextEncoder, self).__init__()
        
        from torchvision import models
        resnet50 = models.resnet50(pretrained=True)
        resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TextDecoder(nn.Module):
    """Decode text from image features"""
    
    def __init__(self, vocab_size, hidden_size=256):
        super(TextDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128 + 256, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, encoder_output, text_indices):
        batch_size = encoder_output.size(0)
        
        # Expand encoder output to sequence length
        encoder_output = encoder_output.unsqueeze(1).expand(-1, text_indices.size(1), -1)
        
        # Embed text indices
        text_embed = self.embedding(text_indices)
        
        # Concatenate encoder output and embedded text
        decoder_input = torch.cat([text_embed, encoder_output], dim=2)
        
        # LSTM
        lstm_output, _ = self.lstm(decoder_input)
        
        # Predict characters
        output = self.fc(lstm_output)
        
        return output


class FullTextOCRModel(nn.Module):
    """Full text OCR model"""
    
    def __init__(self, vocab_size):
        super(FullTextOCRModel, self).__init__()
        self.encoder = TextEncoder()
        self.decoder = TextDecoder(vocab_size)
    
    def forward(self, images, text_indices):
        encoder_output = self.encoder(images)
        output = self.decoder(encoder_output, text_indices)
        return output


def train_ocr_model(data_path, doc_type, output_dir, epochs=20, batch_size=8):
    """Train full-text OCR model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training Full-Text OCR Model - {doc_type.upper()}")
    print(f"{'='*70}")
    print(f"Device: {device}\n")
    
    # Load datasets
    train_dataset = OCRDataset(data_path, doc_type, 'train')
    val_dataset = OCRDataset(data_path, doc_type, 'validation')
    
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")
    
    # Use drop_last=False for validation to avoid empty loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False)
    
    # Model
    model = FullTextOCRModel(train_dataset.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")
    
    # Training
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images, texts, _ = batch
            if images is None or images.shape[0] == 0:
                continue
            
            images = images.unsqueeze(1).to(device)
            texts = texts.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, texts)
            
            # Reshape for loss calculation
            loss = criterion(outputs.reshape(-1, train_dataset.vocab_size), texts.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        if train_batches == 0:
            print(f"Warning: No valid training batches in epoch {epoch+1}")
            continue
        
        avg_train_loss = train_loss / train_batches
        
        # Validate
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for images, texts, _ in val_loader:
                if images is None or images.shape[0] == 0:
                    continue
                
                images = images.unsqueeze(1).to(device)
                texts = texts.to(device)
                
                outputs = model(images, texts)
                loss = criterion(outputs.reshape(-1, train_dataset.vocab_size), texts.reshape(-1))
                val_loss += loss.item()
                val_batches += 1
        
        if val_batches == 0:
            print(f"Warning: No valid validation batches in epoch {epoch+1}")
            avg_val_loss = best_val_loss
        else:
            avg_val_loss = val_loss / val_batches
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{output_dir}/{doc_type}_best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}\n")
                break
        
        scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), f"{output_dir}/{doc_type}_model.pt")
    
    # Save history
    with open(f"{output_dir}/{doc_type}_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save vocabulary
    vocab_data = {
        'char2idx': train_dataset.char2idx,
        'idx2char': {str(k): v for k, v in train_dataset.idx2char.items()},
        'vocab_size': train_dataset.vocab_size
    }
    with open(f"{output_dir}/{doc_type}_vocab.json", 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ TRAINING COMPLETE - {doc_type.upper()}")
    print(f"{'='*70}\n")
    
    return model, train_dataset, history


if __name__ == "__main__":
    try:
        from google.colab import drive
        # In Colab, data is directly accessible
        data_path = "split_data"
        output_dir = "ocr_models"
    except ImportError:
        # Local machine
        data_path = "split_data"
        output_dir = "ocr_models"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("FULL-TEXT OCR MODEL TRAINING")
    print(f"{'='*70}")
    
    # Train prescriptions
    prescr_model, prescr_dataset, prescr_history = train_ocr_model(data_path, 'prescriptions', output_dir, epochs=20)
    
    # Train lab reports
    lab_model, lab_dataset, lab_history = train_ocr_model(data_path, 'lab_reports', output_dir, epochs=20)
    
    print(f"\n{'='*70}")
    print("✓ ALL MODELS TRAINED!")
    print(f"{'='*70}\n")
