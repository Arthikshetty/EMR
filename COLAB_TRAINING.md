# OCR Model Training in Google Colab

## Complete Pipeline: Split → Train → Predict

---

## Step 1: Setup Colab

### 1.1 Create New Notebook
- Go to [colab.research.google.com](https://colab.research.google.com)
- Create new notebook

### 1.2 Enable GPU
```
Runtime → Change runtime type → GPU (Tesla T4)
```

### 1.3 Install Packages
```python
!pip install torch torchvision pillow numpy scikit-learn -q
print("✅ Installed")
```

---

## Step 2: Upload Dataset to Google Drive

### 2.1 Prepare Structure
Upload to `/MyDrive/EMR_Data/`:
```
EMR_Data/
├── data1/
│   ├── Input/    (129 prescription JPG)
│   └── Output/   (129 prescription TXT)
└── lbmaske/
    ├── Input/    (426 lab report PNG)
    └── Output/   (426 lab report TXT)
```

### 2.2 Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
print("✅ Drive mounted")
```

---

## Step 3: Clone Repository

```python
!git clone https://github.com/YOUR_USERNAME/EMR.git /content/EMR
%cd /content/EMR/emr_digitization
print("✅ Repository cloned")
```

---

## Step 4: Split Dataset

```python
exec(open('split_dataset.py').read())
```

**Output:**
```
✓ SPLIT COMPLETE!

PRESCRIPTIONS:
  Train: 91
  Test:  18
  Val:   20

LAB REPORTS:
  Train: 298
  Test:  64
  Val:   64

Output: /content/drive/MyDrive/split_data
```

---

## Step 5: Train Models

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

exec(open('train_model.py').read())
```

**Training Output:**
```
======================================================================
Training PRESCRIPTIONS OCR Model
======================================================================
Device: cuda

Train: 91 | Val: 20 | Test: 18

Epoch  1: Train Loss=2.3421 | Val Loss=1.8934 | Val Acc=45.2%
Epoch  2: Train Loss=1.8743 | Val Loss=1.5621 | Val Acc=52.1%
...
Epoch 20: Train Loss=0.1234 | Val Loss=0.2156 | Val Acc=88.3%

Generating test predictions for prescriptions...
✓ prescriptions predictions saved

======================================================================
Training LAB_REPORTS OCR Model
======================================================================
Device: cuda

Train: 298 | Val: 64 | Test: 64

Epoch  1: Train Loss=2.5123 | Val Loss=2.1034 | Val Acc=42.3%
...
Epoch 20: Train Loss=0.0987 | Val Loss=0.1845 | Val Acc=91.2%

Generating test predictions for lab_reports...
✓ lab_reports predictions saved

======================================================================
✓ ALL MODELS TRAINED!
======================================================================
```

---

## Step 6: Test Models & Generate Predictions

```python
exec(open('test_model.py').read())
```

**Testing Output:**
```
======================================================================
TESTING PRESCRIPTIONS MODEL
======================================================================
Device: cuda

Loading model: ocr_models/prescriptions_best_model.pt
✓ Model loaded

Loading test dataset...
✓ Test samples: 18

Generating predictions for test images...

  file_001.jpg                   → Predicted: 'P' | Actual: 'P' ✓
  file_002.jpg                   → Predicted: 'R' | Actual: 'R' ✓
  file_003.jpg                   → Predicted: 'E' | Actual: 'E' ✓
  ...

======================================================================
✓ TESTING COMPLETE FOR PRESCRIPTIONS
======================================================================

📊 RESULTS:
  Total Test Images: 18
  Correct Predictions: 16
  Accuracy: 88.89%

📁 OUTPUT:
  Predictions (TXT): ocr_models/prescriptions_test_predictions/
  Summary (JSON): ocr_models/prescriptions_test_summary.json
  Details (JSON): ocr_models/prescriptions_test_results.json


======================================================================
TESTING LAB_REPORTS MODEL
======================================================================

  ... (similar output for lab reports)

Accuracy: 89.06%
```

---

## Step 7: Verify Test Results

```python
import json
import os

for doc_type in ['prescriptions', 'lab_reports']:
    # Check summary
    summary_file = f"/content/drive/MyDrive/ocr_models/{doc_type}_test_summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print(f"\n{doc_type.upper()}:")
    print(f"  Accuracy: {summary['accuracy']}")
    print(f"  Correct: {summary['correct_predictions']}/{summary['total_test_samples']}")
    
    # Check prediction files
    pred_dir = f"/content/drive/MyDrive/ocr_models/{doc_type}_test_predictions"
    pred_files = len(os.listdir(pred_dir))
    print(f"  Prediction files: {pred_files}")
```

---

## Step 8: Download Models & Results

```python
# Final models to download
models = [
    '/content/drive/MyDrive/ocr_models/prescriptions_best_model.pt',
    '/content/drive/MyDrive/ocr_models/lab_reports_best_model.pt',
]

# Test predictions (text files)
predictions = [
    '/content/drive/MyDrive/ocr_models/prescriptions_test_predictions/',  # 18 files
    '/content/drive/MyDrive/ocr_models/lab_reports_test_predictions/',    # 64 files
]

# Summary files
summaries = [
    '/content/drive/MyDrive/ocr_models/prescriptions_test_summary.json',
    '/content/drive/MyDrive/ocr_models/lab_reports_test_summary.json',
    '/content/drive/MyDrive/ocr_models/prescriptions_test_results.json',
    '/content/drive/MyDrive/ocr_models/lab_reports_test_results.json',
]

print("Download these files from Google Drive:")
for m in models:
    print(f"  • {m}")
for Final Models (Use These!)
- `prescriptions_best_model.pt` - Best prescription model ✅
- `lab_reports_best_model.pt` - Best lab reports model ✅

### Test Predictions (Individual Text Files)
- `/prescriptions_test_predictions/*.txt` - 18 prediction files (one per test image)
- `/lab_reports_test_predictions/*.txt` - 64 prediction files (one per test image)

**Each file contains:** Single predicted character

### Test Summaries & Details
- `prescriptions_test_summary.json` - Accuracy & results summary
- `lab_reports_test_summary.json` - Accuracy & results summary
- `prescriptions_test_results.json` - Detailed results for each image
- `lab_reports_test_results.json` - Detailed results for each image

### Training Data (Reference)
- `split_data/prescriptions/train|test|validation/` - Split dataset
- `split_data/lab_reports/train|test|validation/` - Split dataset
- `split_summary.json` - Split statistic
- `prescriptions_history.json` - Training metrics
- `lab_reports_history.json` - Training metrics

### Dataset Split
- `split_data/prescriptions/train/` - 91 files
- `split_data/prescriptions/test/` - 18 files
- `split_data/prescriptions/validation/` - 20 files
- `split_data/lab_reports/train/` - 298 files
- `split_data/l Description |
|------|------|-------------|
| Setup | 5 min | Install packages & mount drive |
| Split | 3 min | Split prescriptions & lab reports |
| Train | 30-40 min | Train both models |
| Test | 5 min | Test models & generate predictions |
| **Total** | **~50 min** | Complete pipeline
| Step | Time |
|------|------|
| Setup | 5 min |
| Split Dataset | 3 min |
| Train Prescriptions | 15-20 min |
| Train Lab Reports | 15-20 min |
| **Total** | **~45 min** |

---

## Expected Accuracy

| Model | Val Accuracy |
|-------|-------------|
| Prescriptions | 85-90% |
| Lab Reports | 88-92% |

---

## Quick Copy-Paste for Colab

```python
# 1. Install
!pip install torch torchvision pillow numpy scikit-learn -q

# 2. Mount
from google.colab import drive
drive.mount('/content/drive')

# 3. Clone
!git clone https://github.com/YOUR_USERNAME/EMR.git /content/EMR
%cd /content/EMR/emr_digitization

# 4. Split
exec(open('split_dataset.py').read())

# 5. Train (3 min)
exec(open('split_dataset.py').read())

# 5. Train (30-40 min)
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
exec(open('train_model.py').read())

# 6. Test (5 min) - GENERATES PREDICTION TEXT FILES
exec(open('test_model.py').read())

# 7. Done! Download files from Google Drive
print("✅ Training & Test

## Troubleshooting

**Q: GPU not showing?**
- A: Runtime → Change runtime type → Select GPU

**Q: Files not found?**
- A: Check `/content/drive/MyDrive/EMR_Data/` structure

**Q: Out of memory?**
- A: Reduce batch_size in train_model.py (line ~95)

**Q: Low accuracy?**
- A: Increase epochs to 30-40 in train_model.py

---

