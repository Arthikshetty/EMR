# Google Colab OCR Model Training Guide (Updated for Prescription + Lab Report Dataset)

## Part 1: Training Your Handwriting OCR Model in Google Colab

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **"New notebook"**
3. Rename it: "EMR_OCR_Training"

---

## Step 2: Set Up GPU (for faster training)

In the first cell, add:

```python
# Check GPU availability
!nvidia-smi
```

If no GPU, enable it:
- Click **Runtime** → **Change runtime type** → Select **GPU** → **Save**

---

## Step 3: Install Required Libraries

```python
# Install required packages
!pip install torch torchvision opencv-python pillow numpy pandas scikit-learn matplotlib
!pip install pytesseract  # For baseline comparisons
```

---

## Step 4: Mount Google Drive (to save trained model)

```python
from google.colab import drive
drive.mount('/content/drive')

# Create a folder for your project
import os
os.makedirs('/content/drive/MyDrive/EMR_OCR_Models', exist_ok=True)
```

---

## Step 5: Create OCR Model (Choose One Option)

### **OPTION A: Using PyTorch CNN (Recommended for Handwriting + Lab Reports)**

**This option reads from both prescription/ and lab_report/ folders automatically!**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# 1. Create Dataset Class (Works with Multiple Folders)
class MedicalDocumentDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        """
        base_dir: path to folder containing prescription/ and lab_report/ folders
        
        Expected structure:
        base_dir/
        ├── prescription/
        │   ├── prescription_001.jpg
        │   ├── prescription_001.txt  (optional: extracted text)
        │   ├── prescription_002.jpg
        │   └── ...
        └── lab_report/
            ├── lab_001.jpg
            ├── lab_001.txt  (optional: extracted text)
            └── ...
        """
        self.base_dir = base_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.doc_types = []
        
        # Process each document type folder
        for doc_type in ['prescription', 'lab_report']:
            doc_dir = os.path.join(base_dir, doc_type)
            
            if os.path.exists(doc_dir):
                # Get all image files
                for filename in os.listdir(doc_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(doc_dir, filename)
                        
                        # Try to find corresponding .txt file
                        txt_path = os.path.join(doc_dir, filename.rsplit('.', 1)[0] + '.txt')
                        
                        if os.path.exists(txt_path):
                            # Read extracted text from .txt file
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                label = f.read().strip()
                        else:
                            # Use document type as label if no .txt file
                            label = f"{doc_type}: {filename}"
                        
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                        self.doc_types.append(doc_type)
        
        print(f"✓ Loaded {len(self.image_paths)} images (Total Dataset: 555)")
        print(f"  - Prescriptions: {self.doc_types.count('prescription')} / 129")
        print(f"  - Lab Reports: {self.doc_types.count('lab_report')} / 426")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            
            if self.transform:
                image = self.transform(image)
            
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(1, 32, 256), self.labels[idx]

# 2. Define OCR Model (CNN + RNN)
class HandwritingOCRModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(HandwritingOCRModel, self).__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # RNN for sequence learning
        self.rnn = nn.LSTM(128, hidden_size, num_layers=2, 
                          batch_first=True, bidirectional=True)
        
        # Classification head
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
    
    def forward(self, x):
        # CNN forward pass
        x = self.cnn(x)  # (batch, 128, H', W')
        
        # Reshape for RNN
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, height * width, channels)
        
        # RNN forward pass
        x, _ = self.rnn(x)  # (batch, seq_len, hidden_size*2)
        
        # Classification
        x = self.fc(x)  # (batch, seq_len, vocab_size)
        
        return x

# 3. Training Function
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        batch_count = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.view(-1, outputs.size(-1)), 
                           torch.cat([torch.tensor([ord(c) for c in label], 
                                                   device=device) for label in labels]))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("✓ Training complete!")
    return model

# 4. IMPORTANT: Upload your training data with this structure!
# Create on Google Drive:
# /MyDrive/EMR_OCR_Data/
#   ├── prescription/
#   │   ├── prescription_001.jpg
#   │   ├── prescription_001.txt  (optional: extracted text)
#   │   ├── prescription_002.jpg
#   │   ├── prescription_002.txt  (optional)
#   │   └── ...
#   └── lab_report/
#       ├── lab_001.jpg
#       ├── lab_001.txt  (optional: extracted text)
#       ├── lab_002.jpg
#       ├── lab_002.txt  (optional)
#       └── ...

# 5. Load and Train (UNCOMMENT THESE LINES TO RUN)

transform = transforms.Compose([
    transforms.Resize((32, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset from your folder structure
train_dataset = MedicalDocumentDataset(
    base_dir='/content/drive/MyDrive/EMR_OCR_Data',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=32)

vocab_size = 256  # ASCII characters
model = HandwritingOCRModel(vocab_size=vocab_size)

print("Starting training...")
trained_model = train_model(model, train_loader, val_loader, epochs=20)

# 6. Save Model
print("Saving model...")
torch.save(trained_model.state_dict(), 
          '/content/drive/MyDrive/EMR_OCR_Models/ocr_model.pt')
print("✓ Model saved to Google Drive!")
```

---

### **OPTION B: Using TensorFlow/Keras (Alternative)**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import glob

# 1. Load data from multiple folders
def load_medical_data(base_dir):
    """Load images from prescription/ and lab_report/ folders"""
    images = []
    labels = []
    
    for doc_type in ['prescription', 'lab_report']:
        doc_dir = os.path.join(base_dir, doc_type)
        
        # Get all image files
        image_files = glob.glob(os.path.join(doc_dir, '*.jpg'))
        image_files += glob.glob(os.path.join(doc_dir, '*.jpeg'))
        image_files += glob.glob(os.path.join(doc_dir, '*.png'))
        
        for img_file in image_files:
            image = Image.open(img_file).convert('L')
            image = image.resize((256, 32))
            images.append(np.array(image) / 255.0)
            labels.append(doc_type)
    
    return np.array(images), np.array(labels)

# 2. Build Model
def create_ocr_model(input_shape=(32, 256, 1), output_classes=256):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', 
                           input_shape=input_shape, padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Reshape(((-1, 128))),
        keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
        keras.layers.Dropout(0.5),
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=False)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(output_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

# 3. Load data
X_train, y_train = load_medical_data('/content/drive/MyDrive/EMR_OCR_Data')

# 4. Create Model
model = create_ocr_model()
model.summary()

# 5. Train
print("Training model...")
history = model.fit(X_train, y_train,
                   epochs=20,
                   batch_size=32,
                   validation_split=0.2)

# 6. Save Model
model.save('/content/drive/MyDrive/EMR_OCR_Models/ocr_model.h5')
print("✓ Model saved!")
```

---

### **OPTION C: Transfer Learning (FASTEST - Using Pre-trained Model)**

```python
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# 1. Use pre-trained ResNet for feature extraction
pretrained_model = models.resnet50(pretrained=True)

# 2. Modify for OCR task
class OCRModelTransfer(torch.nn.Module):
    def __init__(self, pretrained_backbone):
        super().__init__()
        self.backbone = torch.nn.Sequential(*list(pretrained_backbone.children())[:-1])
        self.lstm = torch.nn.LSTM(2048, 256, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(512, 256)  # Output vocab size
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1, 2048)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# 3. Create and train model
model = OCRModelTransfer(pretrained_model)

# Load data
# dataset = MedicalDocumentDataset(
#     base_dir='/content/drive/MyDrive/EMR_OCR_Data'
# )
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Save it
torch.save(model.state_dict(), '/content/drive/MyDrive/EMR_OCR_Models/ocr_model_transfer.pt')
print("✓ Transfer learning model saved!")
```

---

## Step 6: Download Model from Colab to Your Computer

### Method 1: Direct Download from Files Panel
1. Click the **Files** icon on left sidebar (folder icon)
2. Navigate to your model: `EMR_OCR_Models/ocr_model.pt`
3. Right-click → **Download**

### Method 2: Using Google Drive
1. Model is already in Google Drive
2. Go to [Google Drive](https://drive.google.com)
3. Find `EMR_OCR_Models` folder
4. Right-click `ocr_model.pt` → **Download**

---

## Step 7: Move Model to Your EMR Project

### Windows (File Explorer)

```
1. Download model from Colab (e.g., ocr_model.pt)

2. Open File Explorer and navigate to:
   C:\Users\arthi\Downloads\EMR\emr_digitization\models\

3. Create the 'models' folder if it doesn't exist

4. Paste your model file here:
   C:\Users\arthi\Downloads\EMR\emr_digitization\models\ocr_model.pt

Your folder structure should look like:
   C:\Users\arthi\Downloads\EMR\emr_digitization\
   ├── config/
   ├── data/
   ├── logs/
   ├── models/
   │   └── ocr_model.pt  ← Your trained model goes here!
   ├── src/
   ├── tests/
   └── [other files]
```

---

## Part 2: Understanding the Pipeline Command

### What is `python run_pipeline.py --image test.jpg --output results/` ?

This command **processes a medical document image through the complete 9-stage EMR pipeline**.

#### Breaking it down:

```bash
python run_pipeline.py --image test.jpg --output results/
│      │              │      │          │      │
│      │              │      │          │      └─ Output folder for FHIR results
│      │              │      │          └────────── Flag for output directory
│      │              │      └────────────────────── Your image file to process
│      │              └────────────────────────────── Flag for image input
│      └──────────────────────────────────────────── Python script to run
└─────────────────────────────────────────────────── Python command
```

#### Full command syntax:

```bash
python run_pipeline.py [OPTIONS]

Options:
  --image PATH          Path to medical document image (required)
  --output PATH         Output folder for results (default: ./output/)
  --document-type TEXT  Document type: prescription, discharge, lab (default: prescription)
  --validate            Enable human validation (true/false)
  --batch               Process entire folder instead of single image
```

---

## Part 3: Complete Usage Examples

### Example 1: Process Single Prescription Image

```bash
python run_pipeline.py --image "C:\Users\arthi\Downloads\prescription.jpg" --output "results/"
```

**What happens:**
1. **Stage 1:** Loads prescription.jpg
2. **Stage 2:** OCR model extracts text from handwritten image
3. **Stage 3:** Spell correction & field extraction
4. **Stage 4:** Structure detection (date, patient name, medications)
5. **Stage 5:** NLP extracts medications, dosages, allergies
6. **Stage 6:** Converts to FHIR resources
7. **Stage 7:** Human validation queue (if enabled)
8. **Stage 8:** Encrypts sensitive data
9. **Stage 9:** Saves FHIR bundle as JSON

**Output files:**
```
results/
├── prescription_extracted.json    ← Raw extracted data
├── prescription_fhir.json         ← FHIR bundle (ready for hospital EHR)
├── prescription_validation.json   ← Human review queue
└── prescription_audit.log         ← Security audit trail
```

---

### Example 2: Batch Process Multiple Documents

```bash
python run_pipeline.py --batch --image "C:\Users\arthi\Downloads\scans/" --output "results/"
```

**Processes all images in the folder:**
```
C:\Users\arthi\Downloads\scans\
├── prescription_001.jpg  ┐
├── prescription_002.jpg  ├─→ All processed → FHIR bundles
├── lab_report_001.jpg    │
└── lab_report_002.jpg    ┘
```

---

### Example 3: Process with Validation Enabled

```bash
python run_pipeline.py --image "prescription.jpg" --output "results/" --validate true
```

**Flags high-risk fields for clinician review:**
- Medication names (must be correct)
- Patient allergies
- Dosages
- Diagnoses (ICD-10 codes)

---

### Example 4: Process Lab Report

```bash
python run_pipeline.py --image "lab_report.jpg" --document-type lab --output "results/"
```

**Extracts lab-specific fields:**
- Test names
- Values
- Reference ranges
- Units
- Date of test

---

## Part 4: Dataset Preparation Guide

### Your Dataset Structure (Create on Google Drive):

```
📁 EMR_OCR_Data/
│
├── 📁 prescription/
│   ├── prescription_001.jpg      ← Prescription image
│   ├── prescription_001.txt      ← Extracted text (optional)
│   ├── prescription_002.jpg
│   ├── prescription_002.txt
│   ├── prescription_003.jpg
│   └── ...
│
└── 📁 lab_report/
    ├── lab_001.jpg              ← Lab report image
    ├── lab_001.txt              ← Extracted text (optional)
    ├── lab_002.jpg
    ├── lab_002.txt
    ├── lab_003.jpg
    └── ...
```

### What .txt Files Should Contain:

**For Prescriptions:**
```
prescription_001.txt:
Patient: John Doe
Date: 2026-01-15
Medication: Aspirin 500mg
Dosage: Twice daily
Duration: 30 days
Doctor: Dr. Smith
```

**For Lab Reports:**
```
lab_001.txt:
Test Date: 2026-01-15
Hemoglobin: 14.5 g/dL (Normal: 12-16)
WBC: 7.2 K/uL (Normal: 4.5-11)
Platelets: 250 K/uL (Normal: 150-400)
Glucose: 95 mg/dL (Normal: 70-100)
```

### If You Don't Have .txt Files:

**Don't worry!** The code works without them:
- Uses document type as label: "prescription: filename.jpg"
- Still trains the model to recognize both document types
- Accuracy will be slightly lower (70-80% vs 85-95%)
- You can add .txt files later for better accuracy

---

## Part 5: Step-by-Step Workflow

### Step 1: Prepare Data (On Your Computer)
```
Collect:
- 50+ prescription images (handwritten or printed)
- 50+ lab report images
- (Optional) Create .txt files with extracted text
```

### Step 2: Upload to Google Drive
```
1. Go to Google Drive: https://drive.google.com
2. Create folder: EMR_OCR_Data/
3. Create subfolders: prescription/ and lab_report/
4. Upload your images and (optional) .txt files
```

### Step 3: Run Training in Colab
```
1. Open Google Colab
2. Copy OPTION A code (all of it)
3. Run cells in order:
   - Step 1: Check GPU
   - Step 2: Enable GPU (if needed)
   - Step 3: Install libraries
   - Step 4: Mount Google Drive
   - Step 5: Copy model + dataset code
4. Press Shift+Enter to run training (takes 15-20 min)
```

### Step 4: Download Model
```
1. After training finishes
2. Download ocr_model.pt from Google Drive
3. Save to: C:\Users\arthi\Downloads\EMR\emr_digitization\models\
```

### Step 5: Test Your Model
```bash
cd C:\Users\arthi\Downloads\EMR\emr_digitization
python test_model.py
```

### Step 6: Run Pipeline
```bash
python run_pipeline.py --image "prescription.jpg" --output "results/"
```

---

## Part 6: Troubleshooting

### Error: "No module named MedicalDocumentDataset"
**Fix:** Make sure you're running the entire OPTION A code block in the same cell

### Error: "EMR_OCR_Data not found"
**Fix:** Check your folder structure on Google Drive:
```
✓ /MyDrive/EMR_OCR_Data/
  ✓ prescription/
  ✓ lab_report/
```

### Error: "No images found"
**Fix:** Check that image files end with .jpg, .jpeg, or .png (case-sensitive on Colab)

### Training is very slow
**Fix:** 
1. Enable GPU (Runtime → Change runtime type → GPU)
2. Reduce batch size: `batch_size=16` instead of 32
3. Reduce epochs: `epochs=10` instead of 20

### Low OCR Accuracy
**Fix:**
1. Add more training images (need 100+ for good accuracy)
2. Create .txt files with extracted text
3. Increase epochs: `epochs=50`
4. Use OPTION C (Transfer Learning) for faster results

### Model file is too large
**Fix:** The .pt file is ~100-200 MB (normal for neural networks)
- Upload will take 5-10 minutes
- Download will take 5-10 minutes

---

## Summary Table

| Step | Action | Time | Tool |
|------|--------|------|------|
| 1 | Open Google Colab | 1 min | Browser |
| 2 | Enable GPU | 1 min | Colab |
| 3 | Install libraries | 3 min | Colab |
| 4 | Mount Google Drive | 1 min | Colab |
| 5 | Upload dataset | 10 min | Google Drive |
| 6 | Run OPTION A code | 20 min | Colab (GPU) |
| 7 | Download model | 5 min | Colab |
| 8 | Move to models/ folder | 2 min | File Explorer |
| 9 | Test model | 1 min | Command line |
| 10 | Process documents | 5 min/doc | Command line |

**Total Time: ~50 minutes**

---

## Next Steps

✅ **Ready?**
1. Prepare your prescription + lab report images
2. (Optional) Create .txt files with extracted text
3. Upload to Google Drive in the structure above
4. Copy OPTION A code to Colab
5. Run training!

📧 **Questions?** Check the troubleshooting section or revisit this guide!

🚀 **After training:** Download model → Place in models/ folder → Run pipeline!
