import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from uuid import uuid4
from enum import Enum

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """FHIR Resource Types"""
    PATIENT = "Patient"
    OBSERVATION = "Observation"
    CONDITION = "Condition"
    MEDICATION = "Medication"
    MEDICATION_REQUEST = "MedicationRequest"
    BUNDLE = "Bundle"


class FHIRValidator:
    """Create and validate FHIR R4 resources"""
    
    def __init__(self, version: str = "R4"):
        self.version = version
        self.base_url = "http://emr.local/fhir"
        logger.info(f"Initialized FHIR Validator for {version}")
    
    def create_patient_resource(self, demographics: Dict[str, Any]) -> Dict:
        """Create FHIR Patient resource"""
        patient_id = str(uuid4())[:8]
        
        patient = {
            "resourceType": ResourceType.PATIENT.value,
            "id": patient_id,
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.utcnow().isoformat() + "Z"
            },
            "identifier": [
                {
                    "use": "usual",
                    "system": f"{self.base_url}/identifier/mrn",
                    "value": demographics.get('patient_id', 'unknown')
                }
            ],
            "name": [
                {
                    "use": "official",
                    "text": demographics.get('patient_name', 'Unknown Patient')
                }
            ],
            "gender": self._normalize_gender(demographics.get('gender')),
            "birthDate": self._parse_date(demographics.get('dob')),
        }
        
        # Add contact if available
        if demographics.get('phone') or demographics.get('email'):
            patient["telecom"] = []
            if demographics.get('phone'):
                patient["telecom"].append({
                    "system": "phone",
                    "value": demographics['phone']
                })
            if demographics.get('email'):
                patient["telecom"].append({
                    "system": "email",
                    "value": demographics['email']
                })
        
        logger.info(f"Created Patient resource: {patient_id}")
        return patient
    
    def create_observation_resource(self, observation_data: Dict[str, Any], 
                                   patient_id: str) -> Dict:
        """Create FHIR Observation resource (for vitals, lab values)"""
        obs_id = str(uuid4())[:8]
        
        observation = {
            "resourceType": ResourceType.OBSERVATION.value,
            "id": obs_id,
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.utcnow().isoformat() + "Z"
            },
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                            "display": "Vital Signs"
                        }
                    ]
                }
            ],
            "code": {
                "text": observation_data.get('display', 'Unknown Observation')
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
            "valueQuantity": {
                "value": observation_data.get('value'),
                "unit": observation_data.get('unit', 'unknown'),
                "system": "http://unitsofmeasure.org"
            }
        }
        
        logger.info(f"Created Observation resource: {obs_id}")
        return observation
    
    def create_condition_resource(self, condition_data: Dict[str, Any],
                                 patient_id: str) -> Dict:
        """Create FHIR Condition resource"""
        cond_id = str(uuid4())[:8]
        
        condition = {
            "resourceType": ResourceType.CONDITION.value,
            "id": cond_id,
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.utcnow().isoformat() + "Z"
            },
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active"
                    }
                ]
            },
            "code": {
                "text": condition_data.get('condition_name', 'Unknown Condition')
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "recordedDate": datetime.utcnow().isoformat() + "Z"
        }
        
        logger.info(f"Created Condition resource: {cond_id}")
        return condition
    
    def create_medication_request(self, medication_data: Dict[str, Any],
                                 patient_id: str) -> Dict:
        """Create FHIR MedicationRequest resource"""
        med_req_id = str(uuid4())[:8]
        
        medication_request = {
            "resourceType": ResourceType.MEDICATION_REQUEST.value,
            "id": med_req_id,
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.utcnow().isoformat() + "Z"
            },
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {
                "text": medication_data.get('medication_name', 'Unknown Medication')
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "authoredOn": datetime.utcnow().isoformat() + "Z",
            "dosageInstruction": [
                {
                    "text": medication_data.get('dosage', 'Not specified'),
                    "timing": {
                        "repeat": {
                            "frequency": medication_data.get('frequency', 1),
                            "period": 1,
                            "periodUnit": "d"
                        }
                    }
                }
            ]
        }
        
        logger.info(f"Created MedicationRequest resource: {med_req_id}")
        return medication_request
    
    def create_fhir_bundle(self, resources: List[Dict]) -> Dict:
        """Create FHIR Bundle containing all resources"""
        bundle_id = str(uuid4())
        
        bundle = {
            "resourceType": ResourceType.BUNDLE.value,
            "id": bundle_id,
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.utcnow().isoformat() + "Z"
            },
            "type": "transaction",
            "entry": [
                {
                    "fullUrl": f"{self.base_url}/{res['resourceType']}/{res['id']}",
                    "resource": res,
                    "request": {
                        "method": "POST" if i > 0 else "PUT",
                        "url": f"{res['resourceType']}/{res.get('id', '')}"
                    }
                }
                for i, res in enumerate(resources)
            ]
        }
        
        logger.info(f"Created FHIR Bundle: {bundle_id} with {len(resources)} resources")
        return bundle
    
    def validate_bundle(self, bundle: Dict) -> bool:
        """Validate FHIR Bundle structure"""
        try:
            # Check required fields
            assert bundle.get('resourceType') == ResourceType.BUNDLE.value
            assert bundle.get('type') in ['transaction', 'batch', 'collection']
            assert isinstance(bundle.get('entry', []), list)
            
            # Validate each resource
            for entry in bundle.get('entry', []):
                resource = entry.get('resource', {})
                assert resource.get('resourceType') in [rt.value for rt in ResourceType]
                assert resource.get('id')
            
            logger.info(f"Bundle {bundle.get('id')} is valid")
            return True
        
        except AssertionError as e:
            logger.error(f"Bundle validation failed: {e}")
            return False
    
    def validate_patient_resource(self, patient: Dict) -> bool:
        """Validate Patient resource"""
        try:
            assert patient.get('resourceType') == ResourceType.PATIENT.value
            assert patient.get('id')
            assert patient.get('name', [])
            logger.info(f"Patient {patient.get('id')} is valid")
            return True
        except AssertionError as e:
            logger.error(f"Patient validation failed: {e}")
            return False
    
    def export_to_json(self, resource: Dict) -> str:
        """Export FHIR resource to JSON string"""
        return json.dumps(resource, indent=2, default=str)
    
    def _normalize_gender(self, gender: Optional[str]) -> Optional[str]:
        """Normalize gender field"""
        if not gender:
            return None
        gender_lower = gender.lower()
        if gender_lower.startswith('m'):
            return 'male'
        elif gender_lower.startswith('f'):
            return 'female'
        elif gender_lower in ['other', 'unknown']:
            return gender_lower
        return 'unknown'
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[str]:
        """Parse date string to FHIR date format (YYYY-MM-DD)"""
        if not date_str:
            return None
        
        import re
        
        # Try different date formats
        patterns = [
            (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', '{0}-{1:02d}-{2:02d}'),  # YYYY-MM-DD
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', '{2}-{0:02d}-{1:02d}'),  # MM-DD-YYYY
        ]
        
        for pattern, fmt in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    groups = match.groups()
                    return fmt.format(*groups)
                except:
                    pass
        
        return None


class HL7FIRConverter:
    """Convert HL7 messages to FHIR (if needed)"""
    
    def __init__(self):
        logger.info("Initialized HL7 to FHIR Converter")
    
    def hl7_to_fhir(self, hl7_message: str) -> Dict:
        """Convert HL7 v2 message to FHIR Bundle"""
        # This is a placeholder for HL7 v2 to FHIR conversion
        # In production, use libraries like HL7apy or similar
        logger.info("Converting HL7 to FHIR")
        return {"resourceType": "Bundle", "type": "transaction", "entry": []}
    
    @staticmethod
    def validate_hl7(hl7_message: str) -> bool:
        """Validate HL7 message format"""
        # Check for HL7 MSH segment
        return hl7_message.startswith('MSH')
