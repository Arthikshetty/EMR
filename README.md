# EMR Digitization System

A robust, AI-driven methodology for converting physical medical records to interoperable FHIR-compliant EMR formats.

## 📊 Current Dataset

- **Total Documents:** 555
- **Prescriptions:** 129 files
- **Lab Reports:** 426 files
- **Training Set:** 444 documents (80%)
- **Validation Set:** 55 documents (10%)
- **Test Set:** 55 documents (10%)
- **Expected Training Time:** 25-30 minutes (with GPU)
- **Expected Accuracy:** 91-95%

## Project Overview

This system addresses key challenges in EMR digitization:

- **Handwritten text recognition** with spell correction
- **Multi-document support** (prescriptions, lab reports)
- **Clinical entity extraction** with medical ontology mapping
- **FHIR R4 conversion** with validation
- **Human-in-the-loop validation** with clinician feedback
- **Active learning** for continuous model improvement
- **HIPAA compliance** with encryption and audit logging
- **Secure deployment** with role-based access control

## Architecture

```
Input (Scanned Medical Documents)
    ↓
1. OCR Extraction [image → text]
    ↓
2. Post-OCR Correction [spell check, abbreviation expansion]
    ↓
3. Structured Field Extraction [demographics, vitals, lab values]
    ↓
4. NLP Entity Extraction [SNOMED CT, ICD-10, LOINC mapping]
    ↓
5. FHIR Resource Creation [Patient, Observation, Condition, etc.]
    ↓
6. Human Validation [clinician review & corrections]
    ↓
7. Active Learning [corrections → retraining]
    ↓
8. Security & Encryption [AES-256, audit logging, HIPAA]
    ↓
Output (FHIR Bundles ready for EHR integration)
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- PostgreSQL 12+ (for production)

### Setup

1. **Clone repository**
```bash
cd emr_digitization
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings
export ENCRYPTION_MASTER_KEY="your-secure-key-here"
```

4. **Place trained OCR model**
```bash
# Copy your trained OCR model to:
mkdir -p models/
cp /path/to/your/ocr_model.pt models/ocr_model.pt
```

5. **Initialize database** (production only)
```bash
python scripts/init_db.py
```

## Usage

### Command Line

**Process single document:**
```bash
python run_pipeline.py \
  --image path/to/document.jpg \
  --doc-type discharge_summary \
  --user-id clinician_001 \
  --output output/
```

**Batch process:**
```bash
python run_pipeline.py \
  --batch-dir path/to/documents/ \
  --doc-type lab_reports \
  --user-id technician_001 \
  --output output/
```

**Get pipeline status:**
```bash
python run_pipeline.py --user-id system
```

### Python API

```python
from src.pipeline import EMRDigitizationPipeline

# Initialize pipeline with trained OCR model
pipeline = EMRDigitizationPipeline(ocr_model_path='models/ocr_model.pt')

# Process single document
result = pipeline.process_document(
    image_path='discharge_summary.jpg',
    document_type='discharge_summary',
    user_id='clinician_001',
    validate=True
)

# Check if successful
if result['success']:
    fhir_bundle = result['fhir_bundle']
    print(f"Created FHIR bundle with {len(fhir_bundle['entry'])} resources")
    
    # Export FHIR bundle
    pipeline.export_fhir_bundle(fhir_bundle, 'output/bundle.json')

# Get pending validations
validations = pipeline.validator.get_pending_validations()

# Submit clinician corrections
pipeline.apply_validation_corrections(
    validation_id=validations[0]['id'],
    corrections={'diagnosis': {'new_value': 'Type 2 Diabetes'}},
    clinician_id='clinician_001'
)
```

### REST API

**Start API server:**
```bash
python api.py
```

**Process document:**
```bash
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/document.jpg",
    "document_type": "discharge_summary",
    "user_id": "clinician_001",
    "validate": true
  }'
```

**Get pending validations:**
```bash
curl http://localhost:8000/api/validation
```

**Submit validation:**
```bash
curl -X POST http://localhost:8000/api/validation/{validation_id} \
  -H "Content-Type: application/json" \
  -d '{
    "clinician_id": "clinician_001",
    "corrections": {
      "diagnosis": {"new_value": "Corrected diagnosis"}
    },
    "notes": "Confirmed by clinician review"
  }'
```

## Configuration

Edit `config/config.yaml` to customize:

- OCR model parameters
- NLP model selection
- FHIR validator settings
- Validation thresholds
- Database connections
- Encryption keys
- API settings

## Key Components

### OCR Module (`src/ocr/`)
- Model wrapper for different OCR frameworks (TensorFlow, PyTorch, ONNX)
- Image preprocessing
- Confidence scoring
- Batch processing

### Post-OCR Processing (`src/post_ocr/`)
- Spell correction with medical dictionary
- Abbreviation expansion
- Structured field extraction (demographics, vitals, labs)

### NLP Module (`src/nlp/`)
- Entity recognition using ClinicalBERT/BioBERT
- Medical ontology mapping (SNOMED CT, ICD-10, LOINC)
- Relationship extraction

### FHIR Conversion (`src/fhir/`)
- FHIR R4 resource creation
- HL7 v2 to FHIR conversion
- Bundle validation
- Interoperability checking

### Validation (`src/validation/`)
- Human-in-loop validation workflow
- High-risk field flagging
- Clinician review interface
- Active learning for model retraining

### Security (`src/security/`)
- AES-256 encryption for sensitive data
- Role-based access control (admin, clinician, technician, auditor)
- HIPAA compliance mechanisms
- Audit logging with tamper detection
- PHI identification and anonymization

## Output Formats

### FHIR Bundle Example
```json
{
  "resourceType": "Bundle",
  "type": "transaction",
  "id": "uuid",
  "timestamp": "2024-01-15T10:30:00Z",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "pat-001",
        "identifier": [{"system": "http://example.com/mrn", "value": "MRN123"}],
        "name": [{"text": "John Doe"}],
        "birthDate": "1980-01-15",
        "gender": "male"
      }
    },
    {
      "resource": {
        "resourceType": "Condition",
        "code": {
          "coding": [{
            "system": "http://hl7.org/fhir/sid/icd-10-cm",
            "code": "E11",
            "display": "Type 2 Diabetes Mellitus"
          }]
        },
        "subject": {"reference": "Patient/pat-001"}
      }
    }
  ]
}
```

## Performance Metrics

Monitor pipeline effectiveness via `/api/status`:

```json
{
  "total_documents": 1250,
  "processed_documents": 1245,
  "failed_documents": 5,
  "validation_pending": 120,
  "validation_metrics": {
    "total_processed": 800,
    "corrections_made": 45,
    "correction_rate": 0.0563,
    "avg_confidence_score": 0.847
  }
}
```

## Validation & Error Handling

- Low-confidence extractions automatically flagged for human review
- High-risk fields (diagnosis, medication, allergies) require validation
- Corrections fed back to active learning pool
- Model automatically retrained when 50+ corrections accumulated

## Security & Compliance

- **Encryption**: AES-256 for data at rest
- **Access Control**: Role-based permissions
- **Audit Logging**: Immutable logs of all data access/modifications
- **PHI Protection**: Identification and special handling of Protected Health Information
- **HIPAA**: Compliant data handling, retention policies, breach notifications

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run integration tests
pytest tests/integration/ -v
```

## Deployment

### Docker
```bash
docker build -t emr-digitization .
docker run -p 8000:8000 \
  -e ENCRYPTION_MASTER_KEY="your-key" \
  -e DB_PASSWORD="your-password" \
  emr-digitization
```

### Kubernetes
```bash
kubectl apply -f k8s/
kubectl logs -f deployment/emr-digitization
```

## Troubleshooting

**OCR Model not loading:**
- Ensure model file exists at configured path
- Check file format matches model type
- Verify CUDA/GPU availability if using GPU

**FHIR Validation failures:**
- Check field data types match FHIR schema
- Ensure required fields are populated
- Review validation error logs

**Performance issues:**
- Enable GPU acceleration in config
- Increase batch size for batch processing
- Use async processing for large documents

## Contributing

1. Create feature branch
2. Add tests for new functionality
3. Ensure all tests pass
4. Submit pull request

## License

MIT License - See LICENSE file

## Contact

For questions or support, contact emr-digitization@example.com

## Roadmap

- [ ] Multi-lingual support (Hindi, Spanish, etc.)
- [ ] Advanced document layout analysis
- [ ] Handwritten signature verification
- [ ] Integration with major EHR systems
- [ ] Mobile app for validation
- [ ] Advanced analytics dashboard
- [ ] Federated learning for distributed training
