#!/usr/bin/env python
"""
QUICK START GUIDE - EMR Pipeline
Run this file to get started with the EMR digitization pipeline
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def check_installation():
    """Check if all required packages are installed"""
    print_header("STEP 1: Checking Installation")
    
    required_packages = [
        'torch',
        'transformers',
        'PIL',
        'yaml',
        'cryptography',
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("  Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages installed!")
        return True

def check_models():
    """Check if trained models exist"""
    print_header("STEP 2: Checking Trained Models")
    
    model_dir = Path("ocr_models")
    
    if not model_dir.exists():
        print(f"  ✗ Model directory not found: {model_dir}")
        print("  Create the directory and add your trained models:")
        print("    - ocr_models/prescriptions_best_model.pt")
        print("    - ocr_models/lab_reports_best_model.pt")
        print("    - ocr_models/prescriptions_vocab.json")
        print("    - ocr_models/lab_reports_vocab.json")
        return False
    
    models = {
        'prescriptions_best_model.pt': 'Prescriptions OCR model',
        'lab_reports_best_model.pt': 'Lab Reports OCR model',
        'prescriptions_vocab.json': 'Prescriptions vocabulary',
        'lab_reports_vocab.json': 'Lab Reports vocabulary',
    }
    
    found = 0
    for model_file, description in models.items():
        path = model_dir / model_file
        if path.exists():
            print(f"  ✓ {description}")
            found += 1
        else:
            print(f"  ✗ {model_file} (MISSING)")
    
    if found == len(models):
        print("\n✓ All models found!")
        return True
    else:
        print(f"\n⚠ {len(models) - found}/{len(models)} models missing")
        return False

def check_data():
    """Check if test data exists"""
    print_header("STEP 3: Checking Test Data")
    
    data_paths = [
        Path("split_data/prescriptions/test"),
        Path("split_data/lab_reports/test"),
    ]
    
    all_exist = True
    for path in data_paths:
        if path.exists():
            image_count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))
            print(f"  ✓ {path} ({image_count} images)")
        else:
            print(f"  ✗ {path} (NOT FOUND)")
            all_exist = False
    
    if all_exist:
        print("\n✓ Test data found!")
        return True
    else:
        print("\n⚠ Some test data missing")
        return False

def run_component_tests():
    """Run component tests"""
    print_header("STEP 4: Running Component Tests")
    
    try:
        import test_components
        print("\n  Running tests...")
        success = test_components.run_all_tests()
        return success
    except Exception as e:
        print(f"  ✗ Tests failed: {e}")
        return False

def show_next_steps(checks):
    """Show next steps based on checks"""
    print_header("NEXT STEPS")
    
    if all(checks.values()):
        print("""
  ✓ System is ready! Run one of these commands:
  
  1. Run complete pipeline (prescriptions):
     python run_pipeline.py --doc-type prescriptions
  
  2. Run complete pipeline (lab reports):
     python run_pipeline.py --doc-type lab_reports
  
  3. Run validation workflow:
     python run_pipeline.py --mode validation
  
  4. Run security audit:
     python run_pipeline.py --mode security
  
  5. Process single document:
     python -c "
from src.pipeline import EMRDigitizationPipeline
p = EMRDigitizationPipeline()
result = p.process_document('image.jpg', 'prescriptions')
print(result)
     "
        """)
    else:
        print("\n  ⚠ System not ready. Fix the following issues:\n")
        if not checks.get('Installation'):
            print("    1. Install missing packages:")
            print("       pip install -r requirements.txt\n")
        
        if not checks.get('Models'):
            print("    2. Add trained models to ocr_models/")
            print("       - Download from Colab training")
            print("       - Or train using train_model.py\n")
        
        if not checks.get('Data'):
            print("    3. Add test data to split_data/")
            print("       - Run split_dataset.py")
            print("       - Or download sample dataset\n")

def main():
    """Main startup check"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  EMR DIGITIZATION PIPELINE - QUICK START GUIDE".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    checks = {}
    
    # Run checks
    checks['Installation'] = check_installation()
    checks['Models'] = check_models()
    checks['Data'] = check_data()
    checks['Tests'] = run_component_tests() if checks['Installation'] else False
    
    # Show results
    print_header("CHECK RESULTS")
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {check_name}")
    
    # Show next steps
    show_next_steps(checks)
    
    print("\n" + "="*80 + "\n")
    
    if all(checks.values()):
        print("✓ System is ready! Happy digitizing! 🎉\n")
        return 0
    else:
        print("⚠ Please fix the issues above before proceeding.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
