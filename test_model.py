"""
Test OCR Model - Generate predictions for test images
Creates individual text files for each test image
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from datetime import datetime


class OCRDataset(Dataset):
    """Load dataset from split folders"""
    
    def __init__(self, data_path, doc_type, split='test'):
        self.image_dir = f"{data_path}/{doc_type}/{split}"
        self.samples = []
        
        if os.path.exists(self.image_dir):
            for f in sorted(os.listdir(self.image_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(self.image_dir, f)
                    txt_name = f.rsplit('.', 1)[0] + '.txt'
                    txt_path = os.path.join(self.image_dir, txt_name)
                    
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as file:
                            text = file.read().strip()
                        if text:
                            self.samples.append((img_path, text, f))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text, filename = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('L')
            image = image.resize((256, 64))
            image = torch.FloatTensor(np.array(image)) / 255.0
            
            # Convert text to class indices (0-255 ASCII)
            text_indices = torch.tensor([min(ord(c), 255) for c in text[:100].ljust(100, ' ')])
            
            return image, text_indices, filename, text
        except:
            return None, None, None, None


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


def test_ocr_model(data_path, model_path, doc_type, output_dir):
    """Test model and generate predictions for each test image"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"TESTING {doc_type.upper()} MODEL")
    print(f"{'='*70}")
    print(f"Device: {device}\n")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = OCRModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded\n")
    
    # Load test dataset
    print(f"Loading test dataset...")
    test_dataset = OCRDataset(data_path, doc_type, 'test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"✓ Test samples: {len(test_dataset)}\n")
    
    # Create output directory for predictions
    pred_dir = f"{output_dir}/{doc_type}_test_predictions"
    os.makedirs(pred_dir, exist_ok=True)
    
    # Test and generate predictions
    print(f"Generating predictions for test images...\n")
    
    all_results = []
    correct_predictions = 0
    
    with torch.no_grad():
        for batch_idx, (images, texts, filenames, actual_texts) in enumerate(test_loader):
            if images is None:
                continue
            
            images = images.unsqueeze(1).to(device)
            texts = texts.to(device)
            
            outputs = model(images)
            predictions = outputs.argmax(1).cpu().numpy()
            
            for pred_char, filename, actual_text in zip(predictions, filenames, actual_texts):
                # Convert prediction to character
                pred_char_ascii = chr(pred_char)
                actual_char = actual_text[0] if actual_text else '?'
                
                # Check if prediction matches
                is_correct = pred_char_ascii == actual_char
                if is_correct:
                    correct_predictions += 1
                
                # Save prediction to individual text file
                pred_file = os.path.join(pred_dir, filename.rsplit('.', 1)[0] + '_prediction.txt')
                with open(pred_file, 'w', encoding='utf-8') as f:
                    f.write(pred_char_ascii)
                
                # Store result
                all_results.append({
                    'filename': filename,
                    'predicted_character': pred_char_ascii,
                    'actual_character': actual_char,
                    'actual_text': actual_text,
                    'is_correct': is_correct,
                    'prediction_file': pred_file
                })
                
                print(f"  {filename:30s} → Predicted: '{pred_char_ascii}' | Actual: '{actual_char}' {'✓' if is_correct else '✗'}")
            
            print(f"Batch {batch_idx + 1}/{len(test_loader)}\n")
    
    # Calculate accuracy
    accuracy = (correct_predictions / len(all_results)) * 100 if all_results else 0
    
    # Save results summary
    summary = {
        'model_type': doc_type,
        'timestamp': datetime.now().isoformat(),
        'total_test_samples': len(all_results),
        'correct_predictions': correct_predictions,
        'accuracy': f"{accuracy:.2f}%",
        'predictions_directory': pred_dir,
        'model_path': model_path
    }
    
    summary_file = f"{output_dir}/{doc_type}_test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    results_file = f"{output_dir}/{doc_type}_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print(f"✓ TESTING COMPLETE FOR {doc_type.upper()}")
    print("="*70)
    print(f"\n📊 RESULTS:")
    print(f"  Total Test Images: {len(all_results)}")
    print(f"  Correct Predictions: {correct_predictions}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"\n📁 OUTPUT:")
    print(f"  Predictions (TXT): {pred_dir}/")
    print(f"  Summary (JSON): {summary_file}")
    print(f"  Details (JSON): {results_file}\n")
    
    return summary, all_results


if __name__ == "__main__":
    try:
        from google.colab import drive
        data_path = "/content/drive/MyDrive/split_data"
        model_dir = "/content/drive/MyDrive/ocr_models"
        output_dir = "/content/drive/MyDrive/ocr_models"
    except:
        data_path = "split_data"
        model_dir = "ocr_models"
        output_dir = "ocr_models"
    
    print(f"\n{'='*70}")
    print("OCR MODEL TESTING")
    print(f"{'='*70}")
    
    # Test prescriptions model
    prescr_model_path = f"{model_dir}/prescriptions_best_model.pt"
    if os.path.exists(prescr_model_path):
        test_ocr_model(data_path, prescr_model_path, 'prescriptions', output_dir)
    else:
        print(f"✗ Prescription model not found: {prescr_model_path}")
    
    # Test lab reports model
    lab_model_path = f"{model_dir}/lab_reports_best_model.pt"
    if os.path.exists(lab_model_path):
        test_ocr_model(data_path, lab_model_path, 'lab_reports', output_dir)
    else:
        print(f"✗ Lab reports model not found: {lab_model_path}")
    
    print(f"\n{'='*70}")
    print("✓ ALL TESTS COMPLETED!")
    print(f"{'='*70}\n")
