#!/usr/bin/env python3
"""
Quick start script for EMR digitization
"""

import sys
from pathlib import Path
import argparse
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import EMRDigitizationPipeline
from src.utils.config import LoggerSetup

def main():
    parser = argparse.ArgumentParser(description='EMR Digitization Pipeline')
    parser.add_argument('--image', type=str, help='Path to single image to process')
    parser.add_argument('--batch-dir', type=str, help='Directory with multiple images')
    parser.add_argument('--ocr-model', type=str, default='./models/ocr_model', help='Path to OCR model')
    parser.add_argument('--output', type=str, help='Output directory for FHIR bundles')
    parser.add_argument('--doc-type', type=str, default='generic', help='Document type')
    parser.add_argument('--user-id', type=str, default='system', help='User ID for audit logging')
    parser.add_argument('--validate', action='store_true', default=True, help='Enable human validation')
    parser.add_argument('--no-validate', dest='validate', action='store_false')
    
    args = parser.parse_args()
    
    # Setup logging
    LoggerSetup.setup()
    
    # Initialize pipeline
    print("Initializing EMR Digitization Pipeline...")
    pipeline = EMRDigitizationPipeline(ocr_model_path=args.ocr_model)
    
    if args.image:
        # Process single document
        print(f"Processing image: {args.image}")
        result = pipeline.process_document(
            image_path=args.image,
            document_type=args.doc_type,
            user_id=args.user_id,
            validate=args.validate
        )
        
        if result['success']:
            print("✓ Document processed successfully")
            
            if args.output:
                output_path = Path(args.output) / f"{Path(args.image).stem}_fhir.json"
                pipeline.export_fhir_bundle(result['fhir_bundle'], str(output_path))
                print(f"✓ FHIR bundle exported to {output_path}")
            
            # Print summary
            print("\n=== Processing Summary ===")
            for stage, data in result['pipeline_stages'].items():
                print(f"{stage}: {data.get('status', 'unknown')}")
        
        else:
            print("✗ Document processing failed")
            for error in result['errors']:
                print(f"  Error: {error}")
    
    elif args.batch_dir:
        # Process batch
        print(f"Processing batch from: {args.batch_dir}")
        results = pipeline.batch_process(
            image_dir=args.batch_dir,
            document_type=args.doc_type,
            user_id=args.user_id
        )
        
        successful = sum(1 for r in results if r['success'])
        print(f"\n✓ Processed {len(results)} documents")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(results) - successful}")
        
        if args.output:
            Path(args.output).mkdir(parents=True, exist_ok=True)
            for result in results:
                if result['success']:
                    output_path = Path(args.output) / f"{Path(result['document_path']).stem}_fhir.json"
                    pipeline.export_fhir_bundle(result['fhir_bundle'], str(output_path))
            print(f"✓ FHIR bundles exported to {args.output}")
    
    else:
        # Print pipeline status
        status = pipeline.get_pipeline_status()
        print("\n=== Pipeline Status ===")
        print(json.dumps(status, indent=2))
    
    print("\n✓ Pipeline execution completed")

if __name__ == '__main__':
    main()
