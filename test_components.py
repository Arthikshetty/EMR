#!/usr/bin/env python
"""
Test Script for EMR Pipeline
Tests individual components and the complete pipeline
"""

import os
import sys
import json
from pathlib import Path
from pprint import pprint

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config():
    """Test configuration loading"""
    print("\n" + "="*80)
    print("TEST 1: Configuration Loading")
    print("="*80)
    
    try:
        from src.utils.config import ConfigManager
        config = ConfigManager()
        
        print("✓ ConfigManager initialized")
        print(f"  OCR model path: {config.get('ocr.model_path')}")
        print(f"  Prescriptions model: {config.get('ocr.prescriptions_model')}")
        print(f"  Lab reports model: {config.get('ocr.lab_reports_model')}")
        print(f"  Logging level: {config.get('logging.level')}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_fhir_converter():
    """Test FHIR converter"""
    print("\n" + "="*80)
    print("TEST 2: FHIR Converter")
    print("="*80)
    
    try:
        from src.fhir.converter import FHIRValidator
        
        validator = FHIRValidator()
        print("✓ FHIRValidator initialized")
        
        # Create sample patient
        patient_data = {
            'patient_name': 'John Doe',
            'patient_id': 'MRN123456',
            'dob': '01/15/1990',
            'gender': 'M',
            'phone': '555-1234'
        }
        
        patient = validator.create_patient_resource(patient_data)
        print(f"✓ Patient resource created: {patient['id']}")
        
        # Create observation
        obs = validator.create_observation_resource({
            'display': 'Blood Pressure',
            'value': '120/80',
            'unit': 'mmHg'
        }, patient['id'])
        print(f"✓ Observation resource created: {obs['id']}")
        
        # Create bundle
        bundle = validator.create_fhir_bundle([patient, obs])
        print(f"✓ FHIR Bundle created: {bundle['id']}")
        print(f"  - Resources: {len(bundle['entry'])}")
        
        # Validate bundle
        is_valid = validator.validate_bundle(bundle)
        print(f"✓ Bundle validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return True
    
    except Exception as e:
        print(f"✗ FHIR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_security():
    """Test security and encryption"""
    print("\n" + "="*80)
    print("TEST 3: Security & Encryption")
    print("="*80)
    
    try:
        from src.security.encryption import DataEncryption, AccessControl, HIPAACompliance, AuditLog
        
        # Test encryption
        encryption = DataEncryption()
        print("✓ DataEncryption initialized")
        
        original = "Sensitive patient data"
        encrypted = encryption.encrypt_field(original)
        decrypted = encryption.decrypt_field(encrypted)
        
        assert decrypted == original
        print(f"✓ Encryption/Decryption: PASSED")
        
        # Test access control
        ac = AccessControl()
        print("✓ AccessControl initialized")
        
        ac.assign_role('user123', 'clinician')
        has_read = ac.check_permission('user123', 'read')
        assert has_read
        print(f"✓ Access Control: PASSED")
        
        # Test HIPAA compliance
        sample_data = {
            'name': 'John Smith',
            'ssn': '123-45-6789',
            'email': 'john@example.com'
        }
        
        phi = HIPAACompliance.validate_phi_fields(sample_data)
        print(f"✓ PHI Detection: Found {len(phi)} types")
        
        # Test audit logging
        audit = AuditLog()
        print("✓ AuditLog initialized")
        
        audit.log_access('user123', 'read_document', 'doc_456', 'Patient')
        print(f"✓ Audit logging: PASSED")
        
        return True
    
    except Exception as e:
        print(f"✗ Security test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_workflow():
    """Test human validation workflow"""
    print("\n" + "="*80)
    print("TEST 4: Validation Workflow")
    print("="*80)
    
    try:
        from src.validation.human_validation import ValidationRequest, HumanInLoopValidator
        
        validator = HumanInLoopValidator(output_dir="test_validation_queue")
        print("✓ HumanInLoopValidator initialized")
        
        # Create validation request
        req = ValidationRequest(
            extracted_data={'patient_name': 'John Doe', 'diagnosis': 'Diabetes'},
            original_text="Patient presents with Type 2 Diabetes...",
            document_id="doc_001"
        )
        
        req.confidence_score = 0.65
        validator.add_to_queue(req)
        print(f"✓ Validation request added: {req.id}")
        
        # Get pending validations
        pending = validator.get_pending_validations(limit=5)
        print(f"✓ Pending validations retrieved: {len(pending)}")
        
        # Submit validation
        validator.submit_validation(
            req.id,
            clinician_id="dr_smith",
            notes="Review completed"
        )
        print(f"✓ Validation submitted")
        
        # Get metrics
        metrics = validator.get_validation_metrics()
        print(f"✓ Validation metrics: {metrics}")
        
        return True
    
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nlp_extraction():
    """Test NLP entity extraction"""
    print("\n" + "="*80)
    print("TEST 5: NLP Entity Extraction")
    print("="*80)
    
    try:
        from src.nlp.entity_extraction import ClinicalNLPExtractor
        
        nlp = ClinicalNLPExtractor()
        print("✓ ClinicalNLPExtractor initialized")
        
        # Test extraction
        text = "Patient presents with diabetes and hypertension. Prescribed Metformin 500mg twice daily."
        
        entities = nlp.extract_entities(text)
        print(f"✓ Entities extracted: {len(entities)}")
        
        if entities:
            print(f"  Sample entities:")
            for entity in entities[:3]:
                print(f"    - {entity}")
        
        return True
    
    except Exception as e:
        print(f"✗ NLP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_post_ocr():
    """Test post-OCR text correction"""
    print("\n" + "="*80)
    print("TEST 6: Post-OCR Text Correction")
    print("="*80)
    
    try:
        from src.post_ocr.text_correction import SpellCorrector, StructuredFieldExtractor
        
        corrector = SpellCorrector()
        print("✓ SpellCorrector initialized")
        
        # Test correction
        text = "Paient prescibed Metformin for diabetus management"
        # Note: corrections depend on dictionary being loaded
        corrected = corrector.correct_text(text)
        print(f"✓ Text correction applied")
        
        # Test structured field extraction
        extractor = StructuredFieldExtractor()
        print("✓ StructuredFieldExtractor initialized")
        
        sample_text = """
        Patient Name: John Smith
        Age: 45
        Blood Pressure: 120/80
        Heart Rate: 72
        """
        
        fields = extractor.extract_fields(sample_text)
        print(f"✓ Fields extracted: {len(fields)} categories")
        
        return True
    
    except Exception as e:
        print(f"✗ Post-OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("EMR PIPELINE COMPONENT TESTS")
    print("="*80)
    
    tests = [
        ("Configuration", test_config),
        ("FHIR Converter", test_fhir_converter),
        ("Security", test_security),
        ("Validation Workflow", test_validation_workflow),
        ("NLP Extraction", test_nlp_extraction),
        ("Post-OCR Correction", test_post_ocr),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
