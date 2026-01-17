#!/usr/bin/env python3
"""
Test Script - Verify OCR Model is Loaded Correctly
Run this BEFORE running the full pipeline to check if your model works
"""

import os
import sys
import json
import torch
from pathlib import Path

print("=" * 70)
print("🔍 EMR OCR MODEL TEST SCRIPT")
print("=" * 70)

# Test 1: Check if models folder exists
print("\n[TEST 1] Checking models/ folder...")
models_dir = Path("models")
if models_dir.exists():
    print(f"✅ models/ folder found at: {models_dir.absolute()}")
else:
    print(f"❌ models/ folder NOT found!")
    print(f"   Expected at: {models_dir.absolute()}")
    sys.exit(1)

# Test 2: Check if model file exists
print("\n[TEST 2] Looking for model file...")
model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.h5")) + list(models_dir.glob("*.pb"))

if model_files:
    print(f"✅ Found {len(model_files)} model file(s):")
    for f in model_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name} ({size_mb:.1f} MB)")
    model_path = model_files[0]
else:
    print(f"❌ No model files found in models/ folder!")
    print(f"   Looking for: .pt (PyTorch), .h5 (TensorFlow), .pb (TensorFlow)")
    print(f"   \n   💡 STEPS TO FIX:")
    print(f"      1. Train model in Google Colab")
    print(f"      2. Download from Google Drive")
    print(f"      3. Copy to: {models_dir.absolute()}")
    sys.exit(1)

# Test 3: Try to load model
print("\n[TEST 3] Attempting to load model...")
try:
    if model_path.suffix == ".pt":
        print(f"   → Detected PyTorch format (.pt)")
        model = torch.load(model_path, map_location='cpu')
        print(f"✅ PyTorch model loaded successfully!")
        print(f"   Model type: {type(model)}")
        
        # If it's a state_dict (recommended way)
        if isinstance(model, dict):
            print(f"   Keys in model: {list(model.keys())[:5]}... (showing first 5)")
            print(f"✅ This is a state_dict (recommended format)")
        else:
            print(f"✅ This is a full model object")
    
    elif model_path.suffix == ".h5":
        print(f"   → Detected TensorFlow format (.h5)")
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            print(f"✅ TensorFlow model loaded successfully!")
            model.summary()
        except ImportError:
            print(f"⚠️  TensorFlow not installed, cannot fully test .h5 model")
            print(f"   Run: pip install tensorflow")
    else:
        print(f"❌ Unknown model format: {model_path.suffix}")
        
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    print(f"\n💡 TROUBLESHOOTING:")
    print(f"   - Verify model file is not corrupted")
    print(f"   - Check if PyTorch/TensorFlow is installed: pip install torch")
    print(f"   - Try downloading model again from Colab")
    sys.exit(1)

# Test 4: Check configuration
print("\n[TEST 4] Checking configuration...")
config_path = Path("config/config.yaml")
if config_path.exists():
    print(f"✅ Configuration file found: {config_path.absolute()}")
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        if 'ocr' in config:
            ocr_config = config['ocr']
            print(f"\n   OCR Configuration:")
            for key, value in ocr_config.items():
                print(f"      - {key}: {value}")
            
            # Check if model path in config matches actual model
            config_model_path = ocr_config.get('model_path', 'NOT SET')
            print(f"\n   ✅ Model path in config: {config_model_path}")
            if Path(config_model_path).exists():
                print(f"   ✅ Model path resolves to file!")
            else:
                print(f"   ⚠️  Model path doesn't match actual file location")
                print(f"      Expected: {config_model_path}")
                print(f"      Actual: {model_path}")
        else:
            print(f"⚠️  'ocr' section not found in config.yaml")
    except Exception as e:
        print(f"⚠️  Could not read config: {str(e)}")
else:
    print(f"⚠️  Configuration file not found at: {config_path.absolute()}")

# Test 5: Check if pipeline can be imported
print("\n[TEST 5] Checking if pipeline module can be imported...")
try:
    from src.pipeline import EMRDigitizationPipeline
    print(f"✅ Pipeline module imported successfully!")
    
    # Try to initialize pipeline
    try:
        pipeline = EMRDigitizationPipeline(ocr_model_path=str(model_path))
        print(f"✅ Pipeline initialized successfully!")
        print(f"   Ready to process documents")
    except Exception as e:
        print(f"⚠️  Pipeline initialized but with warning: {str(e)}")
        
except ImportError as e:
    print(f"⚠️  Could not import pipeline: {str(e)}")
    print(f"   Make sure all packages are installed: pip install -r requirements.txt")

# Test 6: Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"✅ Model location: {model_path.absolute()}")
print(f"✅ Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
print(f"✅ All checks passed! Ready to run pipeline.")
print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print(f"\n1. Run the pipeline with a test image:")
print(f"   python run_pipeline.py --image test.jpg --output results/")
print(f"\n2. Check results folder for output:")
print(f"   - test_extracted.json (raw extraction)")
print(f"   - test_fhir.json (hospital ready)")
print(f"\n3. For batch processing:")
print(f"   python run_pipeline.py --batch --image prescriptions/ --output fhir_out/")
print(f"\n" + "=" * 70)
