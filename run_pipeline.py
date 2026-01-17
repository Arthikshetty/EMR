#!/usr/bin/env python
"""
Complete End-to-End EMR Digitization Pipeline
Processes medical documents through all stages: OCR → Correction → NLP → FHIR → Validation → Security
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import EMRDigitizationPipeline
from src.utils.config import ConfigManager, LoggerSetup

# Setup logging
LoggerSetup.setup()
logger = logging.getLogger(__name__)


def run_complete_pipeline(image_dir: str, doc_type: str = "prescriptions", 
                         validate: bool = True, user_id: str = "admin"):
    """Run complete EMR digitization pipeline"""
    
    print("\n" + "="*80)
    print("EMR DIGITIZATION PIPELINE - COMPLETE WORKFLOW")
    print("="*80 + "\n")
    
    try:
        # Initialize pipeline
        logger.info("Initializing EMR Digitization Pipeline...")
        pipeline = EMRDigitizationPipeline()
        
        # Get image files
        image_dir_path = Path(image_dir)
        if not image_dir_path.exists():
            logger.error(f"Image directory not found: {image_dir}")
            return
        
        image_files = list(image_dir_path.glob("*.jpg")) + list(image_dir_path.glob("*.png"))
        logger.info(f"Found {len(image_files)} images to process")
        
        if not image_files:
            logger.warning("No images found in directory")
            return
        
        # Process first 3 images as demo
        results = []
        for i, image_path in enumerate(image_files[:3]):
            print(f"\n{'='*80}")
            print(f"Processing Document {i+1}: {image_path.name}")
            print(f"{'='*80}")
            
            result = pipeline.process_document(
                str(image_path),
                document_type=doc_type,
                validate=validate,
                user_id=user_id
            )
            
            results.append(result)
            
            # Print summary
            if result.get('success'):
                print("✓ Document processed successfully")
                print(f"  - OCR confidence: {result['pipeline_stages']['ocr'].get('confidence', 'N/A'):.2f}")
                print(f"  - Entities extracted: {result['pipeline_stages']['nlp_extraction'].get('entities_count', 0)}")
                print(f"  - FHIR resources created: {result['pipeline_stages']['fhir_conversion'].get('resources_count', 0)}")
                
                if validate:
                    print(f"  - Validation ID: {result['pipeline_stages']['validation'].get('validation_id')}")
                    print(f"  - Confidence score: {result['pipeline_stages']['validation'].get('confidence_score', 'N/A'):.2f}")
            else:
                print("✗ Document processing failed")
                print(f"  Errors: {result.get('errors')}")
        
        # Print overall statistics
        print(f"\n{'='*80}")
        print("PIPELINE STATISTICS")
        print(f"{'='*80}")
        status = pipeline.get_pipeline_status()
        print(f"Total documents: {status['total_documents']}")
        print(f"Processed: {status['processed_documents']}")
        print(f"Failed: {status['failed_documents']}")
        print(f"Pending validation: {status['validation_pending']}")
        
        if 'validation_metrics' in status:
            print(f"\nValidation Metrics:")
            for key, value in status['validation_metrics'].items():
                print(f"  - {key}: {value}")
        
        # Save results to file
        output_file = Path("pipeline_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Pipeline results saved to {output_file}")
        print(f"\n✓ Pipeline execution completed. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\n✗ Pipeline failed: {e}")


def run_batch_validation():
    """Run batch validation workflow"""
    print("\n" + "="*80)
    print("VALIDATION WORKFLOW")
    print("="*80 + "\n")
    
    try:
        pipeline = EMRDigitizationPipeline()
        
        # Get pending validations
        pending = pipeline.validator.get_pending_validations(limit=10)
        print(f"Pending validations: {len(pending)}")
        
        for i, validation in enumerate(pending[:3]):
            print(f"\n{i+1}. Document: {validation.get('document_id')}")
            print(f"   Confidence: {validation.get('confidence_score', 0):.2f}")
            print(f"   Status: {validation.get('status')}")
            print(f"   High-risk fields: {validation.get('high_risk_fields')}")
            
            # Simulate clinician approval/correction
            corrections = {}
            if validation.get('confidence_score', 1) < 0.75:
                corrections = {
                    'field1': 'corrected_value'
                }
            
            pipeline.validator.submit_validation(
                validation.get('id'),
                clinician_id="dr_smith",
                corrections=corrections if corrections else None,
                notes="Review completed"
            )
        
        # Print validation metrics
        metrics = pipeline.validator.get_validation_metrics()
        print(f"\n{'='*80}")
        print("VALIDATION METRICS")
        print(f"{'='*80}")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Validation workflow failed: {e}")


def run_security_audit():
    """Run security and audit report"""
    print("\n" + "="*80)
    print("SECURITY & COMPLIANCE AUDIT")
    print("="*80 + "\n")
    
    try:
        from src.security.encryption import HIPAACompliance, ComplianceReporter, AuditLog
        
        audit_log = AuditLog()
        
        # Sample audit entries
        sample_data = {
            'patient_name': 'John Doe',
            'ssn': '123-45-6789',
            'dob': '01/15/1990'
        }
        
        # Validate PHI
        phi_fields = HIPAACompliance.validate_phi_fields(sample_data)
        print(f"PHI Fields Identified: {list(phi_fields.keys())}")
        
        # Log access
        audit_log.log_access(
            'clinician_001',
            'access_patient_record',
            'patient_12345',
            'Patient',
            {'phi_fields': list(phi_fields.keys())}
        )
        
        # Generate compliance report
        reporter = ComplianceReporter(audit_log)
        report = reporter.generate_hipaa_report('2026-01-01', '2026-01-31')
        
        print(f"\nHIPAA Compliance Report:")
        print(f"  - Generated: {report.get('generated_at')}")
        print(f"  - Total Accesses: {report.get('total_accesses')}")
        print(f"  - PHI Access Events: {report.get('phi_access_events')}")
        print(f"  - Anomalies Detected: {len(report.get('anomalies_detected', []))}")
        
        # Export report
        reporter.export_compliance_report('compliance_report.json', report)
        print(f"  - Report exported to compliance_report.json")
        
    except Exception as e:
        logger.error(f"Security audit failed: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EMR Digitization Pipeline")
    parser.add_argument('--image-dir', type=str, default='split_data/prescriptions/test',
                       help='Directory containing medical document images')
    parser.add_argument('--doc-type', type=str, default='prescriptions',
                       choices=['prescriptions', 'lab_reports'],
                       help='Document type to process')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Enable human-in-the-loop validation')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'validation', 'security'],
                       help='Execution mode')
    parser.add_argument('--user-id', type=str, default='admin',
                       help='User ID for audit logging')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_complete_pipeline(args.image_dir, args.doc_type, args.validate, args.user_id)
    elif args.mode == 'validation':
        run_batch_validation()
    elif args.mode == 'security':
        run_security_audit()


if __name__ == "__main__":
    main()
