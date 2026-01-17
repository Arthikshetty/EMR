"""
Train Best OCR Model - ResNet50 Transfer Learning
Separate models for Prescriptions and Lab Reports
Generates test predictions with text files
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from datetime import datetime


class OCRDataset(Dataset):
    """Load dataset from split folders"""
    
    def __init__(self, data_path, doc_type, split='train'):
        self.image_dir = f"{data_path}/{doc_type}/{split}"
        self.samples = []
        
        if os.path.exists(self.image_dir):
            for f in os.listdir(self.image_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(self.image_dir, f)
                    txt_name = f.rsplit('.', 1)[0] + '.txt'
                    txt_path = os.path.join(self.image_dir, txt_name)
                    
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as file:
                            text = file.read().strip()
                        if text:
                            self.samples.append((img_path, text))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('L')
            image = image.resize((256, 64))
            image = torch.FloatTensor(np.array(image)) / 255.0
            
            # Convert text to class indices (0-255 ASCII)
            text_indices = torch.tensor([min(ord(c), 255) for c in text[:100].ljust(100, ' ')])
            
            return image, text_indices, os.path.basename(img_path)
        except:
            return None, None, None


class OCRModel(nn.Module):
    """ResNet50-based OCR Model"""
    
    def __init__(self):
        super(OCRModel, self).__init__()
        
        from torchvision import models
        resnet50 = models.resnet50(pretrained=True)
        resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def train_ocr_model(data_path, doc_type, output_dir, epochs=20, batch_size=16):
    """Train OCR model for document type"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training {doc_type.upper()} OCR Model")
    print(f"{'='*70}")
    print(f"Device: {device}\n")
    
    # Load datasets
    train_dataset = OCRDataset(data_path, doc_type, 'train')
    val_dataset = OCRDataset(data_path, doc_type, 'validation')
    test_dataset = OCRDataset(data_path, doc_type, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}\n")
    
    # Model
    model = OCRModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_idx, (images, texts, _) in enumerate(train_loader):
            if images is None:
                continue
            
            images = images.unsqueeze(1).to(device)
            texts = texts.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, texts[:, 0])  # Use first char for simplicity
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for images, texts, _ in val_loader:
                if images is None:
                    continue
                images = images.unsqueeze(1).to(device)
                texts = texts.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, texts[:, 0])
                val_loss += loss.item()
                
                val_acc += (outputs.argmax(1) == texts[:, 0]).float().mean().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader) * 100
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f"Epoch {epoch+1:2d}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={avg_val_acc:.1f}%")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{output_dir}/{doc_type}_best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    # Save final model
    torch.save(model.state_dict(), f"{output_dir}/{doc_type}_model.pt")
    
    # Generate test predictions with text files
    print(f"\nGenerating test predictions for {doc_type}...")
    model.load_state_dict(torch.load(f"{output_dir}/{doc_type}_best_model.pt", map_location=device))
    model.eval()
    
    test_output_dir = f"{output_dir}/{doc_type}_predictions"
    os.makedirs(test_output_dir, exist_ok=True)
    
    predictions = []
    with torch.no_grad():
        for images, texts, names in test_loader:
            if images is None:
                continue
            
            images = images.unsqueeze(1).to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            
            for pred, name, actual_text in zip(preds, names, texts):
                pred_text = chr(pred)
                
                # Save prediction as text file
                pred_file = name.rsplit('.', 1)[0] + '_pred.txt'
                with open(os.path.join(test_output_dir, pred_file), 'w') as f:
                    f.write(pred_text)
                
                predictions.append({
                    'image': name,
                    'predicted_char': pred_text,
                    'actual_char': chr(actual_text[0].item())
                })
    
    # Save predictions
    with open(f"{output_dir}/{doc_type}_predictions.json", 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Save history
    with open(f"{output_dir}/{doc_type}_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✓ {doc_type} predictions saved to {test_output_dir}\n")
    
    return model, history


if __name__ == "__main__":
    try:
        from google.colab import drive
        data_path = "/content/drive/MyDrive/split_data"
        output_dir = "/content/drive/MyDrive/ocr_models"
    except:
        data_path = "split_data"
        output_dir = "ocr_models"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Train both models
    print(f"\n{'='*70}")
    print("TRAINING OCR MODELS")
    print(f"{'='*70}")
    
    train_ocr_model(data_path, 'prescriptions', output_dir, epochs=20)
    train_ocr_model(data_path, 'lab_reports', output_dir, epochs=20)
    
    print(f"\n{'='*70}")
    print("✓ ALL MODELS TRAINED!")
    print(f"{'='*70}")
    print(f"Models saved to: {output_dir}")
    print(f"Predictions saved to: {output_dir}/prescriptions_predictions/")
    print(f"                      {output_dir}/lab_reports_predictions/\n")
