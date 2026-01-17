#!/usr/bin/env python
"""
Pytesseract OCR Test
Verify pytesseract installation and basic functionality
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_pytesseract_installation():
    """Test if pytesseract is properly installed"""
    print("\n" + "="*80)
    print("TEST: Pytesseract Installation")
    print("="*80 + "\n")
    
    try:
        import pytesseract
        print("✓ pytesseract package installed")
        
        # Check if Tesseract-OCR is available
        from PIL import Image
        
        # Try to use pytesseract
        try:
            # This will fail if Tesseract-OCR system package is not installed
            config = pytesseract.get_config()
            print("✓ Tesseract-OCR system package found")
            print(f"  Config: {config[:100]}...")
            return True
        except Exception as e:
            print(f"✗ Tesseract-OCR system package not found")
            print(f"  Error: {e}")
            print("\n  Install Tesseract-OCR:")
            print("  - Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("  - Linux: sudo apt-get install tesseract-ocr")
            print("  - macOS: brew install tesseract")
            return False
    
    except ImportError as e:
        print(f"✗ pytesseract not installed")
        print(f"  Error: {e}")
        print("\n  Install with: pip install pytesseract")
        return False


def test_ocr_wrapper():
    """Test OCR wrapper with pytesseract"""
    print("\n" + "="*80)
    print("TEST: OCR Wrapper")
    print("="*80 + "\n")
    
    try:
        from src.ocr.ocr_wrapper import OCRModelWrapper
        
        ocr = OCRModelWrapper()
        print("✓ OCRModelWrapper initialized with pytesseract")
        
        # Create a simple test image
        from PIL import Image, ImageDraw, ImageFont
        
        test_image_path = "test_ocr.jpg"
        
        # Create simple text image
        img = Image.new('RGB', (400, 100), color='white')
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Patient Name: John Smith", fill='black')
        d.text((10, 40), "Date: 01/15/2026", fill='black')
        d.text((10, 70), "Diagnosis: Diabetes", fill='black')
        img.save(test_image_path)
        print(f"✓ Test image created: {test_image_path}")
        
        # Test OCR
        result = ocr.extract_text(test_image_path)
        
        if result.get('text'):
            print(f"✓ OCR extraction successful")
            print(f"  Text: {result['text'][:100]}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Word count: {result['word_count']}")
            
            # Clean up
            Path(test_image_path).unlink()
            return True
        else:
            print(f"✗ OCR extraction failed: {result.get('error')}")
            Path(test_image_path).unlink()
            return False
    
    except Exception as e:
        print(f"✗ OCR wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading for pytesseract"""
    print("\n" + "="*80)
    print("TEST: Configuration Loading")
    print("="*80 + "\n")
    
    try:
        from src.utils.config import ConfigManager
        
        config = ConfigManager()
        print("✓ ConfigManager initialized")
        
        model_type = config.get('ocr.model_type')
        engine = config.get('ocr.engine')
        
        print(f"  Model type: {model_type}")
        print(f"  Engine: {engine}")
        
        if model_type == "pytesseract":
            print("✓ Config correctly set to pytesseract")
            return True
        else:
            print(f"⚠ Config model type is {model_type}, expected pytesseract")
            return False
    
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_sample_document():
    """Test with actual sample if available"""
    print("\n" + "="*80)
    print("TEST: Sample Document Processing")
    print("="*80 + "\n")
    
    from pathlib import Path
    from src.ocr.ocr_wrapper import OCRModelWrapper
    
    # Look for sample images
    sample_paths = [
        Path("split_data/prescriptions/test"),
        Path("split_data/lab_reports/test"),
        Path("data/data1/Output"),
    ]
    
    for sample_dir in sample_paths:
        if sample_dir.exists():
            images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
            if images:
                print(f"Found sample images in: {sample_dir}")
                print(f"Processing first image: {images[0].name}")
                
                try:
                    ocr = OCRModelWrapper()
                    result = ocr.extract_text(str(images[0]))
                    
                    if result.get('text'):
                        print(f"✓ Successfully processed sample document")
                        print(f"  Confidence: {result['confidence']:.2f}")
                        print(f"  Text preview: {result['text'][:200]}...")
                        return True
                    else:
                        print(f"⚠ OCR returned no text: {result.get('error')}")
                        return False
                except Exception as e:
                    print(f"✗ Failed to process sample: {e}")
                    return False
    
    print("⚠ No sample documents found. Skipping this test.")
    return True


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  PYTESSERACT OCR VERIFICATION".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    tests = [
        ("Installation", test_pytesseract_installation),
        ("Configuration", test_config_loading),
        ("OCR Wrapper", test_ocr_wrapper),
        ("Sample Document", test_with_sample_document),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Pytesseract is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python run_pipeline.py --image-dir split_data/prescriptions/test")
        print("  2. Or: python integration_test.py")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        print("  Main issue: Tesseract-OCR system package not installed")
        print("  Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    print("\n" + "="*80 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
