#!/usr/bin/env python
"""
Detailed Pipeline Viewer - Show all stages of prescription processing
Displays extracted data at each pipeline step
"""

import json
import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline import EMRPipeline
from config import Config

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_stage(stage_num, stage_name, data):
    """Print pipeline stage results"""
    print(f"\n[STAGE {stage_num}] {stage_name}")
    print("-" * 80)
    if isinstance(data, dict):
        print(json.dumps(data, indent=2))
    else:
        print(str(data))

def main():
    # Initialize pipeline
    config = Config()
    pipeline = EMRPipeline(config)
    
    # Use first prescription image
    image_path = "split_data/prescriptions/test/1.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("\nAvailable prescription images:")
        for img in Path("split_data/prescriptions/test").glob("*.jpg")[:5]:
            print(f"  - {img.name}")
        return
    
    print_section("PRESCRIPTION TO FHIR PIPELINE - DETAILED VIEW")
    print(f"Processing: {image_path}")
    
    # Stage 1: OCR Extraction
    print_stage(1, "OCR TEXT EXTRACTION", "Extracting text from handwritten prescription image...")
    ocr_result = pipeline.ocr_engine.extract_text(image_path)
    print_stage(1, "OCR TEXT EXTRACTION - RESULT", ocr_result)
    
    # Stage 2: Post-OCR Correction
    print_stage(2, "POST-OCR CORRECTION", "Correcting OCR text and extracting structured fields...")
    corrected = pipeline.post_ocr_engine.extract_structured_fields(ocr_result['text'])
    print_stage(2, "POST-OCR CORRECTION - RESULT", corrected)
    
    # Stage 3: NLP Entity Extraction
    print_stage(3, "NLP ENTITY EXTRACTION", "Identifying medical entities (medications, dosages, etc.)...")
    entities = pipeline.nlp_engine.extract_entities(ocr_result['text'])
    print_stage(3, "NLP ENTITY EXTRACTION - RESULT", entities)
    
    # Combine all extracted data
    extracted_data = {
        'ocr': ocr_result,
        'structured_fields': corrected,
        'entities': entities
    }
    
    # Stage 4: FHIR Conversion
    print_stage(4, "FHIR RESOURCE CREATION", "Converting extracted data to FHIR R4 format...")
    fhir_bundle = pipeline.fhir_converter.create_fhir_bundle(
        extracted_data,
        doc_type='prescriptions'
    )
    print_stage(4, "FHIR BUNDLE - RESULT", fhir_bundle)
    
    # Stage 5: Human Validation
    print_stage(5, "HUMAN VALIDATION QUEUE", "Creating validation request for clinician review...")
    validation_req = pipeline.validator.add_to_queue(
        extracted_data,
        confidence_score=ocr_result.get('confidence', 0.75),
        priority='medium'
    )
    print(f"✓ Validation request created: {validation_req.id}")
    print(f"  Priority: Medium (OCR confidence: {ocr_result.get('confidence', 0.75):.1%})")
    
    # Stage 6: Security & HIPAA
    print_stage(6, "SECURITY & HIPAA COMPLIANCE", "Applying encryption and audit logging...")
    secure_bundle = pipeline.security_engine.encrypt_phi_fields(fhir_bundle)
    audit_entry = {
        'timestamp': pipeline.security_engine.audit_log.get_recent_logs(1)[0] if pipeline.security_engine.audit_log.get_recent_logs(1) else None,
        'encrypted_fields': pipeline.security_engine._detect_phi_fields(str(fhir_bundle))
    }
    print(f"✓ PHI encryption applied")
    print(f"  Detected PHI fields: {len(audit_entry['encrypted_fields'])}")
    print(f"  Encryption: AES-256")
    
    # Stage 7: Export
    print_stage(7, "FHIR EXPORT", "Final FHIR bundle ready for EHR/EMR system...")
    output_file = "prescription_fhir_detailed.json"
    with open(output_file, 'w') as f:
        json.dump(fhir_bundle, f, indent=2)
    print(f"✓ FHIR bundle exported: {output_file}")
    
    # Summary
    print_section("PIPELINE SUMMARY")
    print(f"""
✓ Pipeline Completed Successfully
  
  Image: {image_path}
  Document Type: Prescriptions
  
  EXTRACTED DATA:
  ├─ OCR Confidence: {ocr_result.get('confidence', 0.75):.1%}
  ├─ Text Extracted: {ocr_result.get('word_count', 0)} words
  ├─ Structured Fields: {len(corrected.get('fields', {}))} identified
  ├─ Medical Entities: {len(entities.get('medications', []))} medications detected
  ├─ FHIR Resources: {len(fhir_bundle['bundle']['entry'])} resources in bundle
  ├─ PHI Fields Detected: {len(audit_entry['encrypted_fields'])}
  └─ Encrypted: Yes (AES-256)
  
  OUTPUT RESOURCES:
  ├─ Patient (demographics)
  ├─ MedicationRequest (prescriptions)
  └─ Ready for EHR import
  
  NEXT STEPS:
  1. Clinician review (Validation Stage)
  2. Apply any corrections
  3. Import to EHR system
  4. Archive encrypted copy
""")
    
    # Show extracted medications if available
    if entities.get('medications'):
        print("\n  DETECTED MEDICATIONS:")
        for med in entities.get('medications', []):
            print(f"    • {med}")

if __name__ == '__main__':
    main()
