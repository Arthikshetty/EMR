# 🏥 EMR DIGITIZATION SYSTEM - Quick Reference Guide

## System Overview

**Goal:** Convert handwritten medical documents (prescriptions, lab reports) → FHIR-compliant electronic medical records

**Current Dataset:** 555 documents (129 prescriptions + 426 lab reports)  
**Training Progress:** 444 documents (80% train) | 55 validation | 55 test  
**Input:** Scanned JPG/PNG of handwritten document  
**Output:** FHIR Bundle JSON ready for hospital EHR integration

---

## 9-Stage Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HANDWRITTEN DOCUMENT                               │
│              (Prescription, Discharge Summary, Lab Report)                  │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ↓
        ┌────────────────────────────────────────┐
        │  STAGE 1: IMAGE PREPROCESSING          │
        │  - Load image (300+ DPI)               │
        │  - Convert to RGB                      │
        │  - Normalize pixels                    │
        │  - Resize to optimal size              │
        └────────────┬─────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────────────────────┐
        │  STAGE 2: OCR EXTRACTION               │
        │  [YOUR TRAINED MODEL]                  │
        │  - Extract text from handwriting       │
        │  - Generate confidence score           │
        │  - Return predictions                  │
        └────────────┬─────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────────────────────┐
        │  STAGE 3: POST-OCR CORRECTION          │
        │  - Spell checking                      │
        │  - Abbreviation expansion              │
        │  - Medication name fixing              │
        │  - Medical terminology normalization   │
        └────────────┬─────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────────────────────┐
        │  STAGE 4: STRUCTURED EXTRACTION        │
        │  - Demographics (name, DOB, ID, gender)│
        │  - Vitals (BP, HR, Temp, RR, O2Sat)    │
        │  - Lab Values (glucose, WBC, etc.)     │
        └────────────┬─────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────────────────────┐
        │  STAGE 5: NLP ENTITY EXTRACTION        │
        │  - Extract clinical entities           │
        │  - Map to medical standards:           │
        │    • SNOMED CT (diagnoses)             │
        │    • ICD-10 (diagnosis codes)          │
        │    • RxNorm (medications)              │
        │    • LOINC (lab tests)                 │
        │  - Extract relationships               │
        └────────────┬─────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────────────────────┐
        │  STAGE 6: FHIR CONVERSION              │
        │  - Create Patient resource             │
        │  - Create Observation resources        │
        │  - Create Condition resources          │
        │  - Create MedicationRequest resources  │
        │  - Bundle into transaction             │
        └────────────┬─────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────────────────────┐
        │  STAGE 7: HUMAN VALIDATION             │
        │  - Check confidence level              │
        │  - Flag high-risk fields               │
        │  - Queue for clinician review          │
        │  - Collect corrections                 │
        └────────────┬─────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────────────────────┐
        │  STAGE 8: SECURITY & ENCRYPTION        │
        │  - Identify PHI (Protected Health Info)│
        │  - Apply AES-256 encryption            │
        │  - Create audit trail                  │
        │  - Log user access                     │
        │  - Enforce HIPAA compliance            │
        └────────────┬─────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────────────────────┐
        │  STAGE 9: ACTIVE LEARNING              │
        │  - Store clinician corrections         │
        │  - When 50+ samples: create batch      │
        │  - Retrain OCR + NLP models            │
        │  - Improve accuracy for next docs      │
        └────────────┬─────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FHIR BUNDLE (JSON)                                  │
│                    Ready for Hospital EHR Import                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components (8 Modules)

### 1️⃣ **OCR Module** (`src/ocr/ocr_wrapper.py`)
- Wraps your trained handwriting recognition model
- Supports: PyTorch (.pt), TensorFlow (.h5, .pb), ONNX, pickle
- Functions:
  - `preprocess_image()` - Image preparation
  - `extract_text()` - Run inference & get results
  - `batch_extract_text()` - Process multiple images

```python
ocr = OCRModelWrapper('models/ocr_model.pt')
result = ocr.extract_text('prescription.jpg')
# Output: {'text': '...', 'confidence': 0.85}
```

### 2️⃣ **Post-OCR Module** (`src/post_ocr/text_correction.py`)
- **SpellCorrector**: Fixes OCR errors using medical dictionary
- **FieldExtractor**: Isolates structured medical data

```python
corrector = SpellCorrector('config/medical_dictionary.json')
corrected = corrector.correct_text("Diabeties BP 130/80 q.i.d.")
# Output: "Diabetes blood pressure 130/80 four times daily"
```

### 3️⃣ **NLP Module** (`src/nlp/entity_extraction.py`)
- Extracts clinical entities (conditions, medications, etc.)
- Maps to medical standards (SNOMED CT, ICD-10, RxNorm, LOINC)
- Identifies relationships (medication + dosage)

```python
nlp = ClinicalNLPExtractor()
entities = nlp.extract_entities("Patient has diabetes, prescribed amoxicillin")
normalized = nlp.normalize_entities(entities)
# Output: [{entity: 'diabetes', icd10: 'E11', snomed_ct: '73211009'}, ...]
```

### 4️⃣ **FHIR Module** (`src/fhir/converter.py`)
- Creates FHIR R4 resources from extracted data
- Validates FHIR structure
- Supports HL7 v2 to FHIR conversion

```python
fhir = FHIRValidator()
patient = fhir.create_patient_resource(demographics)
condition = fhir.create_condition_resource(diagnosis, patient_id)
bundle = fhir.create_fhir_bundle([patient, condition, ...])
```

### 5️⃣ **Validation Module** (`src/validation/human_validation.py`)
- **HumanInLoopValidator**: Manages clinician review workflows
- **ActiveLearningManager**: Collects corrections for retraining

```python
validator = HumanInLoopValidator()
validator.add_to_queue(validation_request)
validations = validator.get_pending_validations()
validator.submit_validation(validation_id, corrections={'dosage': {...}})
```

### 6️⃣ **Security Module** (`src/security/encryption.py`)
- **DataEncryption**: AES-256 for sensitive data
- **AccessControl**: Role-based permissions
- **AuditLog**: Immutable audit trail
- **HIPAACompliance**: Healthcare compliance

```python
encryptor = DataEncryption()
encrypted = encryptor.encrypt_data(patient_data)

access_control = AccessControl()
access_control.assign_role('user_001', 'clinician')
can_access = access_control.check_permission('user_001', 'write')
```

### 7️⃣ **Configuration Module** (`src/utils/config.py`)
- Loads YAML configuration
- Manages medical dictionary
- Centralizes all settings

```python
config = ConfigManager()
ocr_threshold = config.get('ocr.confidence_threshold')  # 0.7
```

### 8️⃣ **Pipeline Orchestrator** (`src/pipeline.py`)
- Chains all modules together
- Coordinates 9-stage processing
- Error handling & logging

```python
pipeline = EMRDigitizationPipeline(ocr_model_path='models/ocr_model.pt')
result = pipeline.process_document('prescription.jpg', 'prescription')
# Returns complete processing result with all stages
```

---

## Data Flow Example: Prescription Processing

### Input (Handwritten Prescription):
```
John Doe, 15/01/1980
Diabeties
Amoxycillin 500mg q.i.d.
BP 130/80
```

### Stage 1-3 (OCR → Correction):
```
OCR Output: "John Doe, 15/01/1980, Diabeties, Amoxycillin 500mg q.i.d., BP 130/80"
↓
Corrected: "John Doe, 15/01/1980, Diabetes, Amoxicillin 500mg four times daily, BP 130/80"
```

### Stage 4 (Structured Extraction):
```json
{
  "demographics": {
    "patient_name": "John Doe",
    "dob": "1980-01-15"
  },
  "vitals": {
    "blood_pressure": "130/80"
  }
}
```

### Stage 5 (NLP Normalization):
```json
{
  "diagnoses": [
    {
      "text": "Diabetes",
      "icd10": "E11",
      "snomed_ct": "73211009"
    }
  ],
  "medications": [
    {
      "name": "Amoxicillin",
      "dosage": "500mg",
      "frequency": "four times daily",
      "rxnorm": "309096"
    }
  ]
}
```

### Stage 6 (FHIR Bundle):
```json
{
  "resourceType": "Bundle",
  "type": "transaction",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "name": [{"text": "John Doe"}],
        "birthDate": "1980-01-15"
      }
    },
    {
      "resource": {
        "resourceType": "Condition",
        "code": {"coding": [{"code": "E11", "display": "Type 2 Diabetes"}]}
      }
    },
    {
      "resource": {
        "resourceType": "MedicationRequest",
        "medicationReference": {"display": "Amoxicillin"},
        "dosageInstruction": [{"text": "500mg four times daily"}]
      }
    }
  ]
}
```

---

## Quick Start Commands

### 1. Initialize Environment:
```powershell
# Install dependencies
pip install -r requirements.txt

# Place your OCR model
cp your_model.pt models/ocr_model.pt
```

### 2. Process Single Document:
```powershell
python run_pipeline.py `
  --image "path/to/prescription.jpg" `
  --doc-type prescription `
  --user-id doctor_001 `
  --output "output/" `
  --validate
```

### 3. Batch Processing:
```powershell
python run_pipeline.py `
  --batch-dir "path/to/documents/" `
  --doc-type lab_reports `
  --user-id technician_001 `
  --output "output/"
```

### 4. Start REST API:
```powershell
python api.py
# Access at http://localhost:8000
```

### 5. Python API Usage:
```python
from src.pipeline import EMRDigitizationPipeline

pipeline = EMRDigitizationPipeline(ocr_model_path='models/ocr_model.pt')

# Process document
result = pipeline.process_document(
    image_path='prescription.jpg',
    document_type='prescription',
    user_id='doctor_001',
    validate=True
)

# Export FHIR
pipeline.export_fhir_bundle(result['fhir_bundle'], 'output/bundle.json')

# Get status
status = pipeline.get_pipeline_status()
print(status)
```

---

## REST API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/process` | POST | Process single document |
| `/api/batch-process` | POST | Process multiple documents |
| `/api/validation` | GET | Get pending validations |
| `/api/validation/{id}` | POST | Submit validation corrections |
| `/api/status` | GET | Get pipeline metrics |
| `/api/export-fhir` | POST | Export FHIR bundle |
| `/health` | GET | Health check |
| `/config` | GET | Get configuration |

---

## Configuration (config.yaml)

```yaml
ocr:
  model_path: "models/ocr_model"
  confidence_threshold: 0.7

post_ocr:
  medical_dictionary: "config/medical_dictionary.json"
  confidence_threshold: 0.75

nlp:
  model_name: "allenai/scibert_scivocab_uncased"
  confidence_threshold: 0.7

fhir:
  version: "R4"

validation:
  human_in_loop: true
  auto_learning: true

security:
  encryption_algorithm: "AES-256"
  hipaa_compliance: true
```

---

## Key Features Summary

✅ **Handles Handwritten Documents**
- Trained OCR model integration
- Spell correction for OCR errors
- Medical terminology normalization

✅ **Multiple Document Types**
- Prescriptions (medication extraction)
- Discharge summaries (comprehensive records)
- Lab reports (test results)

✅ **Medical Standards Compliance**
- FHIR R4 output
- ICD-10 diagnosis codes
- RxNorm medication codes
- LOINC lab test codes
- SNOMED CT terminology

✅ **Quality Assurance**
- Human-in-loop validation
- High-risk field flagging
- Clinician review workflows
- Confidence scoring

✅ **Continuous Improvement**
- Active learning from corrections
- Automatic model retraining
- Confidence threshold tuning

✅ **Enterprise Security**
- AES-256 encryption
- HIPAA compliance
- Audit logging
- Role-based access control
- PHI protection

---

## Workflow Overview

```
1. SCAN           → Handwritten medical document
2. UPLOAD         → To system
3. OCR            → Extract text from handwriting
4. CORRECT        → Fix spelling & abbreviations
5. STRUCTURE      → Extract key fields
6. NORMALIZE      → Map to medical standards
7. CONVERT        → Create FHIR resources
8. VALIDATE       → Clinician review (if needed)
9. SECURE         → Encrypt & audit
10. IMPROVE       → Use corrections to retrain
11. EXPORT        → FHIR bundle for EHR import
12. INTEGRATE     → Hospital systems ingest directly
```

---

## Success Metrics

Monitor via `/api/status` endpoint:

```json
{
  "total_documents": 1250,
  "processed_documents": 1245,
  "failed_documents": 5,
  "validation_pending": 50,
  "validation_metrics": {
    "total_processed": 1100,
    "corrections_made": 85,
    "correction_rate": 0.077,
    "avg_confidence_score": 0.87
  },
  "active_learning_pool_size": 35,
  "training_batches": 2
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OCR model not loading | Check file exists at path, verify format (.pt/.h5/.onnx) |
| FHIR validation errors | Ensure all required fields populated, check data types |
| Low confidence scores | Check image quality (DPI >300), handwriting clarity |
| Performance issues | Enable GPU, increase batch size, use async processing |
| Validation queue growing | Reduce confidence threshold or increase clinicians |

---

## Next Steps

1. **Upload your trained OCR model** to `models/` folder
2. **Test with sample documents** from `data/` folder
3. **Monitor validations** via REST API
4. **Collect clinician feedback** for continuous improvement
5. **Retrain models** using active learning batches
6. **Deploy to production** when confidence reaches target (>90%)

---

**Status:** ✅ Ready to integrate your trained OCR model  
**Next Action:** Upload model → Test pipeline → Deploy
