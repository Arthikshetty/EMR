#!/usr/bin/env python3
"""
Demo: Show FHIR output for handwritten prescriptions & lab reports
Uses simulated OCR to demonstrate the complete pipeline
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.post_ocr.text_correction import StructuredFieldExtractor
from src.nlp.entity_extraction import ClinicalNLPExtractor
from src.fhir.converter import FHIRValidator
from src.security.encryption import DataEncryption, AuditLog, HIPAACompliance
from src.validation.human_validation import ValidationRequest, HumanInLoopValidator


# Sample handwritten prescription text (what OCR would extract)
SAMPLE_PRESCRIPTION = """
Patient: Maria Garcia
Age: 52
MRN: MG987654321
DOB: 05/12/1971
Gender: Female

Rx Date: 01/15/2026

Medication: Lisinopril
Dose: 10mg
Form: Tablet
Qty: 30
Sig: One tablet daily
Refills: 3

Medication: Atorvastatin
Dose: 20mg
Form: Tablet
Qty: 30
Sig: One tablet at bedtime
Refills: 6

Diagnosis: Hypertension, Type 2 Diabetes
Prescriber: Dr. James Wilson, MD
License: MD456789
"""

# Sample handwritten lab report text
SAMPLE_LAB_REPORT = """
LABORATORY RESULTS REPORT

Patient: Robert Johnson
Age: 68
MRN: RJ555888
DOB: 03/22/1957
Gender: Male

Test Date: 01/14/2026

CHEMISTRY PANEL:
Glucose: 145 mg/dL (Reference: 70-100)
BUN: 22 mg/dL (Reference: 7-20)
Creatinine: 1.2 mg/dL (Reference: 0.7-1.3)
Sodium: 138 mEq/L (Reference: 135-145)
Potassium: 4.1 mEq/L (Reference: 3.5-5.0)

LIPID PANEL:
Total Cholesterol: 215 mg/dL (Reference: <200)
HDL: 38 mg/dL (Reference: >40)
LDL: 145 mg/dL (Reference: <100)
Triglycerides: 180 mg/dL (Reference: <150)

Lab Director: Dr. Sarah Chen, MD
"""


def process_sample_document(sample_text, doc_type, output_file=None):
    """
    Process sample handwritten document and generate FHIR
    
    Args:
        sample_text: OCR extracted text
        doc_type: 'prescriptions' or 'lab_reports'
        output_file: Where to save FHIR
    """
    
    print("\n" + "="*80)
    print(f"PROCESSING HANDWRITTEN {doc_type.upper()} - DEMO")
    print("="*80 + "\n")
    
    # Stage 1: Simulate OCR with confidence
    print("[1/8] OCR Extraction (Simulated)")
    ocr_confidence = 0.85
    print(f"  ✓ Text extracted (confidence: {ocr_confidence:.0%})")
    print(f"  ✓ Characters: {len(sample_text)}")
    
    # Stage 2: Post-OCR Processing
    print("\n[2/8] Post-OCR Correction...")
    extractor = StructuredFieldExtractor()
    structured = extractor.extract_fields(sample_text)
    print(f"  ✓ Structured fields extracted:")
    for category, fields in structured.items():
        if fields:
            print(f"    - {category}: {len(fields)} fields")
            for field, value in list(fields.items())[:2]:
                if value:
                    print(f"      • {field}: {value}")
    
    # Stage 3: NLP Entity Extraction
    print("\n[3/8] NLP Entity Extraction...")
    nlp = ClinicalNLPExtractor()
    entities = nlp.extract_entities(sample_text)
    print(f"  ✓ Medical entities found: {len(entities)}")
    for entity in entities[:3]:
        if isinstance(entity, dict):
            print(f"    - {entity.get('word', entity)}: {entity.get('entity', 'ENTITY')}")
        else:
            print(f"    - {entity}")
    
    # Stage 4: FHIR Bundle Creation
    print("\n[4/8] FHIR Bundle Creation...")
    fhir = FHIRValidator()
    
    # Create Patient
    patient_data = structured.get('demographics', {})
    patient = fhir.create_patient_resource(patient_data)
    resources = [patient]
    print(f"  ✓ Patient resource created: {patient['id']}")
    print(f"    - Name: {patient_data.get('patient_name', 'Unknown')}")
    print(f"    - MRN: {patient_data.get('patient_id', 'N/A')}")
    
    # Create medication or observation resources
    if doc_type == 'prescriptions':
        if structured.get('medications'):
            for med_name, med_details in structured['medications'].items():
                if med_details:
                    med_req = fhir.create_medication_request(med_details, patient['id'])
                    resources.append(med_req)
                    print(f"  ✓ Medication: {med_name}")
    
    elif doc_type == 'lab_reports':
        if structured.get('lab_values'):
            for test_name, test_value in structured['lab_values'].items():
                if test_value:
                    obs = fhir.create_observation_resource({
                        'display': test_name,
                        'value': test_value,
                        'unit': 'unknown'
                    }, patient['id'])
                    resources.append(obs)
                    print(f"  ✓ Lab Value: {test_name}")
    
    # Create Bundle
    bundle = fhir.create_fhir_bundle(resources)
    is_valid = fhir.validate_bundle(bundle)
    print(f"  ✓ FHIR Bundle created: {bundle['id']}")
    print(f"  ✓ Resources in bundle: {len(resources)}")
    print(f"  ✓ Validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Stage 5: Human Validation
    print("\n[5/8] Validation Workflow...")
    validator = HumanInLoopValidator(output_dir="handwritten_demo")
    validation_req = ValidationRequest(
        extracted_data=structured,
        original_text=sample_text,
        document_id=f"{doc_type}_demo"
    )
    validation_req.confidence_score = ocr_confidence
    validator.add_to_queue(validation_req)
    validator.submit_validation(
        validation_req.id,
        clinician_id="dr_demo",
        corrections=None,
        notes="Demo validation - APPROVED"
    )
    print(f"  ✓ Validation created: {validation_req.id}")
    print(f"  ✓ Status: Approved by dr_demo")
    
    # Stage 6: Security
    print("\n[6/8] Security & HIPAA Compliance...")
    phi = HIPAACompliance.validate_phi_fields(patient_data)
    encryption = DataEncryption()
    encrypted_data = encryption.encrypt_dict(patient_data, list(phi.keys()))
    print(f"  ✓ PHI fields detected: {len(phi)}")
    for phi_type, fields in phi.items():
        print(f"    - {phi_type}: {fields}")
    print(f"  ✓ Data encrypted with AES-256")
    
    # Stage 7: Audit
    print("\n[7/8] Audit Logging...")
    audit = AuditLog(log_file="handwritten_demo/audit.log")
    audit.log_access(
        user_id="demo_user",
        action="process_handwritten",
        resource_id=f"{doc_type}_demo",
        resource_type=doc_type,
        details={"ocr_confidence": ocr_confidence, "entities": len(entities)}
    )
    print(f"  ✓ Audit trail created")
    
    # Stage 8: Export FHIR
    print("\n[8/8] Export FHIR Bundle...")
    
    fhir_output = {
        "bundle": bundle,
        "metadata": {
            "document_id": f"{doc_type}_demo",
            "document_type": doc_type,
            "processing_timestamp": datetime.utcnow().isoformat(),
            "ocr_confidence": ocr_confidence,
            "validation_id": validation_req.id,
            "validator": "dr_demo",
            "entities_extracted": len(entities),
            "phi_fields": len(phi),
            "resources_created": len(resources)
        }
    }
    
    if not output_file:
        output_file = f"handwritten_demo_{doc_type}_fhir.json"
    
    with open(output_file, 'w') as f:
        json.dump(fhir_output, f, indent=2, default=str)
    
    print(f"  ✓ FHIR Bundle exported: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    
    summary = {
        "Status": "✓ SUCCESS",
        "DocumentType": doc_type,
        "OCRConfidence": f"{ocr_confidence:.0%}",
        "PatientName": patient_data.get('patient_name', 'Unknown'),
        "PatientMRN": patient_data.get('patient_id', 'N/A'),
        "EntitiesExtracted": len(entities),
        "ResourcesCreated": len(resources),
        "PHIFieldsDetected": len(phi),
        "ValidationStatus": "Approved",
        "Encryption": "AES-256",
        "FHIRBundle": bundle['id'],
        "OutputFile": output_file
    }
    
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ FHIR output saved to: {output_file}\n")
    
    return fhir_output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo: Process handwritten medical documents to FHIR")
    parser.add_argument("--doc-type", choices=['prescriptions', 'lab_reports'], default='prescriptions',
                       help="Document type to process")
    parser.add_argument("--output", help="Output file (default: handwritten_demo_[type]_fhir.json)")
    
    args = parser.parse_args()
    
    try:
        # Choose sample based on type
        if args.doc_type == 'prescriptions':
            sample_text = SAMPLE_PRESCRIPTION
        else:
            sample_text = SAMPLE_LAB_REPORT
        
        # Process
        result = process_sample_document(
            sample_text=sample_text,
            doc_type=args.doc_type,
            output_file=args.output
        )
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
