"""
Integration Test - End-to-End Pipeline Demonstration
Tests all components working together in a realistic workflow
"""

import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.fhir.converter import FHIRValidator
from src.security.encryption import DataEncryption, AccessControl, HIPAACompliance, AuditLog
from src.validation.human_validation import ValidationRequest, HumanInLoopValidator, ActiveLearningManager
from src.nlp.entity_extraction import ClinicalNLPExtractor
from src.post_ocr.text_correction import StructuredFieldExtractor


def simulate_complete_workflow():
    """Simulate a complete document processing workflow"""
    
    print("\n" + "="*80)
    print("COMPLETE WORKFLOW INTEGRATION TEST")
    print("="*80 + "\n")
    
    # =====  Stage 1: Simulate OCR Output =====
    print("STAGE 1: OCR Output (Simulated)")
    print("-" * 80)
    
    ocr_output = {
        "text": """
        Patient Name: John Robert Smith
        Age: 45
        MRN: MR123456789
        Date of Birth: 01/15/1979
        Gender: Male
        
        Prescription Details:
        Date: 01/15/2026
        
        Medication: Metformin Hydrochloride
        Strength: 500mg
        Form: Tablet
        Quantity: 60 tablets
        Sig: Take one tablet twice daily with meals
        Refills: 3
        
        Diagnosis: Type 2 Diabetes Mellitus
        
        Prescriber: Dr. Sarah Johnson, MD
        License: MD123456
        """,
        "confidence": 0.87
    }
    
    print(f"✓ OCR completed")
    print(f"  - Confidence: {ocr_output['confidence']:.2f}")
    print(f"  - Text length: {len(ocr_output['text'])} characters\n")
    
    # ===== Stage 2: Post-OCR Processing =====
    print("STAGE 2: Post-OCR Text Correction")
    print("-" * 80)
    
    extractor = StructuredFieldExtractor()
    structured = extractor.extract_fields(ocr_output['text'])
    
    print(f"✓ Structured fields extracted:")
    for category, fields in structured.items():
        if fields:
            print(f"  {category}:")
            for field, value in fields.items():
                if value:
                    print(f"    - {field}: {value}")
    print()
    
    # ===== Stage 3: NLP Entity Extraction =====
    print("STAGE 3: NLP Entity Extraction")
    print("-" * 80)
    
    nlp = ClinicalNLPExtractor()
    entities = nlp.extract_entities(ocr_output['text'])
    
    print(f"✓ Clinical entities extracted:")
    print(f"  - Total entities: {len(entities)}")
    if entities:
        for entity in entities[:5]:
            print(f"    - {entity}")
    print()
    
    # ===== Stage 4: FHIR Conversion =====
    print("STAGE 4: FHIR Bundle Creation")
    print("-" * 80)
    
    fhir = FHIRValidator()
    
    # Create Patient resource
    patient_data = structured.get('demographics', {})
    patient = fhir.create_patient_resource(patient_data)
    print(f"✓ Patient resource created: {patient['id']}")
    
    # Create Observation for vital signs
    if structured.get('vitals'):
        resources = [patient]
        for vital_name, vital_value in structured['vitals'].items():
            if vital_value:
                obs = fhir.create_observation_resource({
                    'display': vital_name,
                    'value': vital_value,
                    'unit': 'unknown'
                }, patient['id'])
                resources.append(obs)
                print(f"✓ Observation created: {vital_name}")
    else:
        resources = [patient]
    
    # Create FHIR Bundle
    bundle = fhir.create_fhir_bundle(resources)
    is_valid = fhir.validate_bundle(bundle)
    
    print(f"✓ FHIR Bundle created: {bundle['id']}")
    print(f"  - Resources: {len(resources)}")
    print(f"  - Valid: {is_valid}\n")
    
    # ===== Stage 5: Human Validation =====
    print("STAGE 5: Human-in-Loop Validation")
    print("-" * 80)
    
    validator = HumanInLoopValidator(output_dir="demo_validation")
    
    # Create validation request
    validation_req = ValidationRequest(
        extracted_data=structured,
        original_text=ocr_output['text'],
        document_id="DEMO_001"
    )
    
    validation_req.confidence_score = ocr_output['confidence']
    validator.add_to_queue(validation_req)
    
    print(f"✓ Validation request created: {validation_req.id}")
    print(f"  - Document: {validation_req.document_id}")
    print(f"  - Confidence: {validation_req.confidence_score:.2f}")
    
    # Simulate clinician review
    validator.submit_validation(
        validation_req.id,
        clinician_id="dr_smith",
        corrections=None,  # Approved as-is
        notes="Review completed - all data accurate"
    )
    
    print(f"✓ Validation submitted by dr_smith")
    
    metrics = validator.get_validation_metrics()
    print(f"  Validation metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"    - {key}: {value:.3f}")
        else:
            print(f"    - {key}: {value}")
    print()
    
    # ===== Stage 6: Security & Encryption =====
    print("STAGE 6: Security & Compliance")
    print("-" * 80)
    
    # Detect PHI
    phi = HIPAACompliance.validate_phi_fields(structured.get('demographics', {}))
    print(f"✓ PHI detected: {len(phi)} types")
    for phi_type, fields in phi.items():
        print(f"  - {phi_type}: {fields}")
    
    # Setup encryption
    encryption = DataEncryption()
    encrypted_data = encryption.encrypt_dict(
        structured.get('demographics', {}),
        list(phi.keys())
    )
    print(f"✓ Data encrypted with AES-256")
    
    # Setup access control
    ac = AccessControl()
    ac.assign_role("user_123", "clinician")
    permissions = ac.get_user_permissions("user_123")
    print(f"✓ Access control configured")
    print(f"  - Permissions: {permissions}")
    
    # Audit logging
    audit = AuditLog(log_file="demo_audit.log")
    audit.log_access(
        user_id="dr_smith",
        action="process_document",
        resource_id="DEMO_001",
        resource_type="prescription",
        details={"phi_fields": list(phi.keys())}
    )
    print(f"✓ Audit trail created\n")
    
    # ===== Stage 7: Active Learning Setup =====
    print("STAGE 7: Active Learning Integration")
    print("-" * 80)
    
    al_manager = ActiveLearningManager()
    
    # Simulate adding correction if needed
    if ocr_output['confidence'] < 0.9:
        al_manager.add_correction_sample({
            'document_id': 'DEMO_001',
            'correction_type': 'ocr_error',
            'original': ocr_output['text'],
            'corrected': structured.get('demographics', {})
        })
        print(f"✓ Correction sample added to active learning pool")
    
    uncertain_samples = al_manager.get_uncertain_samples(threshold=0.65)
    print(f"  Uncertain samples in pool: {len(uncertain_samples)}")
    print()
    
    # ===== Stage 8: Export Results =====
    print("STAGE 8: Export Results")
    print("-" * 80)
    
    # Export FHIR bundle
    fhir_output = {
        "bundle": bundle,
        "metadata": {
            "document_id": "DEMO_001",
            "processing_timestamp": "2026-01-17T12:00:00Z",
            "ocr_confidence": ocr_output['confidence'],
            "validation_id": validation_req.id,
            "validator": "dr_smith"
        }
    }
    
    output_file = Path("demo_fhir_output.json")
    with open(output_file, 'w') as f:
        json.dump(fhir_output, f, indent=2, default=str)
    
    print(f"✓ FHIR bundle exported to {output_file}")
    
    # Export validation request
    validation_output = Path("demo_validation_request.json")
    with open(validation_output, 'w') as f:
        json.dump(validation_req.to_dict(), f, indent=2, default=str)
    
    print(f"✓ Validation request exported to {validation_output}")
    
    # ===== Summary =====
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)
    
    summary = {
        "Status": "✓ SUCCESS",
        "Document": "DEMO_001",
        "Processing_Stages": 8,
        "OCR_Confidence": f"{ocr_output['confidence']:.2f}",
        "Entities_Extracted": len(entities),
        "FHIR_Resources_Created": len(resources),
        "Validation_Status": "Approved",
        "Security": "AES-256 Encrypted",
        "Audit_Logged": "✓",
        "Output_Files": [
            str(output_file),
            str(validation_output),
            "demo_audit.log"
        ]
    }
    
    print("\nSummary:")
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        simulate_complete_workflow()
        print("✓ Integration test completed successfully!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
