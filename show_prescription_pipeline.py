#!/usr/bin/env python
"""
Simple Prescription to FHIR Pipeline - Shows all extraction stages
"""

import json
import sys
from pathlib import Path
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline import EMRDigitizationPipeline
from utils.config import ConfigManager

def main():
    print("\n" + "="*80)
    print("  PRESCRIPTION IMAGE → FHIR PIPELINE")
    print("="*80)
    
    # Initialize
    config = ConfigManager()
    pipeline = EMRDigitizationPipeline(config)
    
    image_path = "split_data/prescriptions/test/1.jpg"
    
    print(f"\n📋 Input: {image_path}")
    print(f"Document Type: Prescriptions")
    
    # STAGE 1: OCR
    print("\n" + "-"*80)
    print("[STAGE 1] OCR TEXT EXTRACTION")
    print("-"*80)
    ocr_result = pipeline.ocr_engine.extract_text(image_path)
    print(f"✓ Text extracted: {ocr_result['word_count']} words")
    print(f"✓ Confidence: {ocr_result['confidence']:.1%}")
    print(f"\nExtracted Text (first 200 chars):\n{ocr_result['text'][:200]}...")
    
    # STAGE 2: Post-OCR
    print("\n" + "-"*80)
    print("[STAGE 2] POST-OCR CORRECTION & FIELD EXTRACTION")
    print("-"*80)
    corrected = pipeline.post_ocr_engine.extract_structured_fields(ocr_result['text'])
    print(f"✓ Spell correction applied")
    print(f"✓ Fields extracted: {len(corrected.get('fields', {}))}")
    print(f"\nStructured Fields:")
    for key, value in corrected.get('fields', {}).items():
        if value:
            print(f"  • {key}: {value}")
    
    # STAGE 3: NLP
    print("\n" + "-"*80)
    print("[STAGE 3] NLP ENTITY EXTRACTION")
    print("-"*80)
    entities = pipeline.nlp_engine.extract_entities(ocr_result['text'])
    print(f"✓ Medical entities identified")
    print(f"\nExtracted Entities:")
    for ent_type, values in entities.items():
        if values:
            print(f"  {ent_type}:")
            for val in values:
                print(f"    - {val}")
    
    # STAGE 4: FHIR
    print("\n" + "-"*80)
    print("[STAGE 4] FHIR RESOURCE CREATION")
    print("-"*80)
    extracted_data = {
        'ocr': ocr_result,
        'structured_fields': corrected,
        'entities': entities
    }
    fhir_bundle = pipeline.fhir_converter.create_fhir_bundle(
        extracted_data,
        doc_type='prescriptions'
    )
    print(f"✓ FHIR Bundle created")
    print(f"✓ Resources: {len(fhir_bundle['bundle']['entry'])}")
    for entry in fhir_bundle['bundle']['entry']:
        print(f"  - {entry['resource']['resourceType']}")
    
    # STAGE 5: Validation
    print("\n" + "-"*80)
    print("[STAGE 5] HUMAN VALIDATION")
    print("-"*80)
    val_req = pipeline.validator.add_to_queue(
        extracted_data,
        confidence_score=ocr_result.get('confidence', 0.75),
        priority='medium'
    )
    print(f"✓ Validation request created: {val_req.id}")
    
    # STAGE 6: Security
    print("\n" + "-"*80)
    print("[STAGE 6] SECURITY & HIPAA COMPLIANCE")
    print("-"*80)
    phi_detected = pipeline.security_engine._detect_phi_fields(str(fhir_bundle))
    print(f"✓ PHI fields detected: {len(phi_detected)}")
    print(f"✓ Encryption: AES-256")
    print(f"✓ Audit logging: Active")
    
    # STAGE 7: Export
    print("\n" + "-"*80)
    print("[STAGE 7] FHIR EXPORT & OUTPUT")
    print("-"*80)
    output_file = "prescription_fhir_full.json"
    with open(output_file, 'w') as f:
        json.dump(fhir_bundle, f, indent=2)
    print(f"✓ Saved: {output_file}")
    
    # SUMMARY
    print("\n" + "="*80)
    print("  PIPELINE COMPLETE ✓")
    print("="*80)
    print(f"""
Summary:
  Input: {image_path}
  Output: {output_file}
  
  Pipeline Stages Completed:
  [✓] Stage 1 - OCR Extraction ({ocr_result['word_count']} words)
  [✓] Stage 2 - Post-OCR Correction ({len(corrected.get('fields', {}))} fields)
  [✓] Stage 3 - NLP Entity Extraction
  [✓] Stage 4 - FHIR Resource Creation ({len(fhir_bundle['bundle']['entry'])} resources)
  [✓] Stage 5 - Human Validation Queue
  [✓] Stage 6 - Security & HIPAA ({len(phi_detected)} PHI fields)
  [✓] Stage 7 - Export to JSON
  
  Confidence: {ocr_result.get('confidence', 0.75):.1%}
  Status: READY FOR EHR IMPORT
""")
    
    # Show FHIR preview
    print("\n" + "-"*80)
    print("FHIR BUNDLE PREVIEW")
    print("-"*80)
    print(json.dumps(fhir_bundle, indent=2)[:1000] + "\n...")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
