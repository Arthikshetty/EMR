#!/usr/bin/env python3
"""
Process Handwritten Prescriptions & Lab Reports
Converts handwritten medical documents to FHIR format
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import EMRDigitizationPipeline
from src.ocr.ocr_wrapper import OCRModelWrapper
from src.post_ocr.text_correction import StructuredFieldExtractor
from src.nlp.entity_extraction import ClinicalNLPExtractor
from src.fhir.converter import FHIRValidator
from src.security.encryption import DataEncryption, AuditLog, HIPAACompliance
from src.validation.human_validation import ValidationRequest, HumanInLoopValidator


def process_handwritten_documents(image_dir, doc_type, output_dir="handwritten_output"):
    """
    Process handwritten medical documents
    
    Args:
        image_dir: Directory with handwritten images
        doc_type: 'prescriptions' or 'lab_reports'
        output_dir: Where to save FHIR outputs
    
    Returns:
        List of FHIR bundles
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"✗ Directory not found: {image_dir}")
        return []
    
    # Find all images
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    
    if not images:
        print(f"✗ No images found in {image_dir}")
        return []
    
    print("\n" + "="*80)
    print(f"PROCESSING {len(images)} HANDWRITTEN {doc_type.upper()}")
    print("="*80 + "\n")
    
    # Initialize components
    ocr = OCRModelWrapper()
    spell_corrector = StructuredFieldExtractor()
    nlp = ClinicalNLPExtractor()
    fhir = FHIRValidator()
    validator = HumanInLoopValidator(output_dir=str(output_path / "validations"))
    encryption = DataEncryption()
    audit = AuditLog(log_file=str(output_path / "audit.log"))
    
    results = []
    
    for idx, image_path in enumerate(images[:10], 1):  # Process first 10
        print("="*80)
        print(f"Document {idx}: {image_path.name}")
        print("="*80)
        
        try:
            # Stage 1: OCR
            print("\n[1/8] OCR Text Extraction...")
            ocr_result = ocr.extract_text(str(image_path))
            
            if not ocr_result or not ocr_result.get('text'):
                print(f"  ✗ OCR failed - could not extract text")
                continue
            
            ocr_text = ocr_result['text']
            ocr_confidence = ocr_result.get('confidence', 0.0)
            
            print(f"  ✓ Text extracted (confidence: {ocr_confidence:.2%})")
            print(f"  ✓ Text length: {len(ocr_text)} characters")
            
            # Stage 2: Post-OCR Processing
            print("\n[2/8] Post-OCR Correction...")
            structured = spell_corrector.extract_fields(ocr_text)
            print(f"  ✓ Fields extracted: {len(structured)} categories")
            for category in structured:
                if structured[category]:
                    print(f"    - {category}: {len(structured[category])} fields")
            
            # Stage 3: NLP Entity Extraction
            print("\n[3/8] NLP Entity Extraction...")
            entities = nlp.extract_entities(ocr_text)
            print(f"  ✓ Entities found: {len(entities)}")
            for entity in entities[:3]:
                print(f"    - {entity}")
            
            # Stage 4: FHIR Conversion
            print("\n[4/8] FHIR Bundle Creation...")
            
            # Create Patient resource
            patient_data = structured.get('demographics', {})
            patient = fhir.create_patient_resource(patient_data)
            resources = [patient]
            print(f"  ✓ Patient resource created: {patient['id']}")
            
            # Create other resources based on document type
            if doc_type == 'prescriptions' and structured.get('medications'):
                for med in structured['medications'].values():
                    if med:
                        med_req = fhir.create_medication_request(med, patient['id'])
                        resources.append(med_req)
                        print(f"  ✓ Medication request created")
            
            elif doc_type == 'lab_reports' and structured.get('lab_values'):
                for lab_name, lab_value in structured['lab_values'].items():
                    if lab_value:
                        obs = fhir.create_observation_resource({
                            'display': lab_name,
                            'value': lab_value,
                            'unit': 'unknown'
                        }, patient['id'])
                        resources.append(obs)
                        print(f"  ✓ Observation created: {lab_name}")
            
            # Create bundle
            bundle = fhir.create_fhir_bundle(resources)
            is_valid = fhir.validate_bundle(bundle)
            print(f"  ✓ FHIR Bundle created: {bundle['id']}")
            print(f"  ✓ Bundle valid: {is_valid}")
            
            # Stage 5: Human Validation
            print("\n[5/8] Validation Request...")
            validation_req = ValidationRequest(
                extracted_data=structured,
                original_text=ocr_text,
                document_id=image_path.stem
            )
            validation_req.confidence_score = ocr_confidence
            validator.add_to_queue(validation_req)
            print(f"  ✓ Validation request created: {validation_req.id}")
            
            # Stage 6: Security & HIPAA
            print("\n[6/8] Security & Encryption...")
            phi = HIPAACompliance.validate_phi_fields(patient_data)
            print(f"  ✓ PHI fields detected: {len(phi)}")
            
            encrypted_data = encryption.encrypt_dict(patient_data, list(phi.keys()))
            print(f"  ✓ Data encrypted with AES-256")
            
            # Stage 7: Audit Logging
            print("\n[7/8] Audit Trail...")
            audit.log_access(
                user_id="system",
                action="process_handwritten_document",
                resource_id=image_path.stem,
                resource_type=doc_type,
                details={"ocr_confidence": ocr_confidence, "entities": len(entities)}
            )
            print(f"  ✓ Audit logged")
            
            # Stage 8: Export Results
            print("\n[8/8] Export FHIR...")
            
            fhir_output = {
                "bundle": bundle,
                "metadata": {
                    "document_id": image_path.stem,
                    "document_type": doc_type,
                    "source_image": str(image_path.name),
                    "processing_timestamp": datetime.utcnow().isoformat(),
                    "ocr_confidence": ocr_confidence,
                    "validation_id": validation_req.id,
                    "entities_extracted": len(entities),
                    "phi_fields_detected": len(phi)
                }
            }
            
            output_file = output_path / f"{image_path.stem}_fhir.json"
            with open(output_file, 'w') as f:
                json.dump(fhir_output, f, indent=2, default=str)
            
            print(f"  ✓ FHIR exported to: {output_file.name}")
            
            results.append({
                'success': True,
                'document_id': image_path.stem,
                'fhir_file': str(output_file),
                'bundle_id': bundle['id'],
                'confidence': ocr_confidence
            })
            
            print(f"\n✓ Document processed successfully!\n")
            
        except Exception as e:
            print(f"\n✗ Error processing {image_path.name}: {e}\n")
            results.append({
                'success': False,
                'document_id': image_path.stem,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    
    print(f"\nProcessed: {len(results)} documents")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    
    print(f"\nOutput directory: {output_path}")
    print("\nGenerated files:")
    for result in results:
        if result['success']:
            print(f"  ✓ {result['document_id']}_fhir.json")
    
    # Save summary
    summary_file = output_path / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_documents': len(results),
            'successful': successful,
            'failed': failed,
            'timestamp': datetime.utcnow().isoformat(),
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Summary saved to: processing_summary.json\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process handwritten medical documents")
    parser.add_argument("--image-dir", required=True, help="Directory with images")
    parser.add_argument("--doc-type", choices=['prescriptions', 'lab_reports'], required=True,
                       help="Document type")
    parser.add_argument("--output-dir", default="handwritten_output", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        results = process_handwritten_documents(
            image_dir=args.image_dir,
            doc_type=args.doc_type,
            output_dir=args.output_dir
        )
        
        successful = sum(1 for r in results if r['success'])
        if successful > 0:
            print("✓ Processing completed successfully!")
            sys.exit(0)
        else:
            print("✗ No documents processed successfully")
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
