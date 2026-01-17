"""
EMR Digitization API Server
Flask REST API for document processing and validation
"""

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import logging
from pathlib import Path
import json

from src.pipeline import EMRDigitizationPipeline
from src.validation.human_validation import HumanInLoopValidator
from src.security.encryption import AccessControl

app = Flask(__name__)
api = Api(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pipeline
pipeline = None
access_control = AccessControl()

def init_pipeline(ocr_model_path: str = None):
    """Initialize the EMR digitization pipeline"""
    global pipeline
    pipeline = EMRDigitizationPipeline(ocr_model_path=ocr_model_path)
    logger.info("Pipeline initialized")

class DocumentProcessing(Resource):
    """Process single document endpoint"""
    
    def post(self):
        """
        Process a medical document
        
        Request JSON:
        {
            "image_path": "/path/to/document.jpg",
            "document_type": "discharge_summary",
            "user_id": "clinician_001",
            "validate": true
        }
        """
        try:
            # Check authorization
            user_id = request.json.get('user_id')
            if not access_control.check_permission(user_id, 'write'):
                return {'error': 'Unauthorized'}, 403
            
            image_path = request.json.get('image_path')
            document_type = request.json.get('document_type', 'generic')
            validate = request.json.get('validate', True)
            
            if not image_path or not Path(image_path).exists():
                return {'error': 'Invalid image path'}, 400
            
            # Process document
            result = pipeline.process_document(
                image_path=image_path,
                document_type=document_type,
                user_id=user_id,
                validate=validate
            )
            
            return {'result': result}, 200
        
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {'error': str(e)}, 500

class BatchProcessing(Resource):
    """Batch document processing endpoint"""
    
    def post(self):
        """
        Process multiple documents from directory
        
        Request JSON:
        {
            "image_dir": "/path/to/documents",
            "document_type": "lab_reports",
            "user_id": "technician_001"
        }
        """
        try:
            user_id = request.json.get('user_id')
            if not access_control.check_permission(user_id, 'write'):
                return {'error': 'Unauthorized'}, 403
            
            image_dir = request.json.get('image_dir')
            document_type = request.json.get('document_type', 'generic')
            
            if not Path(image_dir).is_dir():
                return {'error': 'Invalid directory'}, 400
            
            results = pipeline.batch_process(image_dir, document_type, user_id)
            
            return {
                'total_documents': len(results),
                'successful': sum(1 for r in results if r['success']),
                'failed': sum(1 for r in results if not r['success']),
                'results': results
            }, 200
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return {'error': str(e)}, 500

class ValidationManagement(Resource):
    """Manage human validation workflow"""
    
    def get(self, validation_id=None):
        """Get pending validations or specific validation"""
        try:
            if validation_id:
                # Get specific validation
                validations = pipeline.validator.get_pending_validations(100)
                validation = next((v for v in validations if v['id'] == validation_id), None)
                return validation or {'error': 'Not found'}, 404
            else:
                # Get pending validations
                validations = pipeline.validator.get_pending_validations()
                return {
                    'pending_count': len(validations),
                    'validations': validations
                }, 200
        
        except Exception as e:
            logger.error(f"Validation retrieval error: {e}")
            return {'error': str(e)}, 500
    
    def post(self, validation_id):
        """Submit validation with corrections"""
        try:
            clinician_id = request.json.get('clinician_id')
            corrections = request.json.get('corrections', {})
            notes = request.json.get('notes', '')
            
            success = pipeline.apply_validation_corrections(
                validation_id, corrections, clinician_id
            )
            
            return {
                'success': success,
                'message': 'Validation submitted' if success else 'Validation not found'
            }, 200 if success else 404
        
        except Exception as e:
            logger.error(f"Validation submission error: {e}")
            return {'error': str(e)}, 500

class PipelineStatus(Resource):
    """Get pipeline status and metrics"""
    
    def get(self):
        """Get overall pipeline statistics"""
        try:
            status = pipeline.get_pipeline_status()
            return {'status': status}, 200
        except Exception as e:
            logger.error(f"Status retrieval error: {e}")
            return {'error': str(e)}, 500

class FHIRExport(Resource):
    """Export FHIR bundles"""
    
    def post(self):
        """
        Export FHIR bundle to file
        
        Request JSON:
        {
            "fhir_bundle": {...},
            "output_path": "/path/to/output.json"
        }
        """
        try:
            fhir_bundle = request.json.get('fhir_bundle')
            output_path = request.json.get('output_path')
            
            success = pipeline.export_fhir_bundle(fhir_bundle, output_path)
            
            return {
                'success': success,
                'output_path': output_path if success else None
            }, 200 if success else 500
        
        except Exception as e:
            logger.error(f"Export error: {e}")
            return {'error': str(e)}, 500

# Register resources
api.add_resource(DocumentProcessing, '/api/process')
api.add_resource(BatchProcessing, '/api/batch-process')
api.add_resource(ValidationManagement, '/api/validation', '/api/validation/<validation_id>')
api.add_resource(PipelineStatus, '/api/status')
api.add_resource(FHIRExport, '/api/export-fhir')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {'status': 'healthy', 'pipeline': pipeline is not None}, 200

@app.route('/config', methods=['GET'])
def get_config():
    """Get pipeline configuration (non-sensitive)"""
    try:
        from src.utils.config import ConfigManager
        config = ConfigManager()
        return {
            'ocr_confidence_threshold': config.get('ocr.confidence_threshold'),
            'nlp_confidence_threshold': config.get('nlp.confidence_threshold'),
            'fhir_version': config.get('fhir.version'),
            'human_validation_enabled': config.get('validation.human_in_loop')
        }, 200
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    # Initialize with OCR model
    ocr_model_path = './models/ocr_model'  # Update with actual model path
    init_pipeline(ocr_model_path)
    
    # Run server
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True
    )
