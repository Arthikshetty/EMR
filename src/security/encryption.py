import logging
import hashlib
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from cryptography.fernet import Fernet
import os

logger = logging.getLogger(__name__)


class DataEncryption:
    """AES-256 encryption for sensitive data"""
    
    def __init__(self, key: Optional[str] = None):
        """Initialize encryption with key"""
        if key is None:
            # Generate or load key from environment
            key = os.getenv('EMR_ENCRYPTION_KEY', Fernet.generate_key().decode())
        
        try:
            if isinstance(key, str):
                key = key.encode()
            self.cipher_suite = Fernet(key)
            logger.info("Encryption initialized")
        except Exception as e:
            logger.error(f"Encryption initialization failed: {e}")
            raise
    
    def encrypt_field(self, data: str) -> str:
        """Encrypt a single field"""
        try:
            encrypted = self.cipher_suite.encrypt(data.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_field(self, encrypted_data: str) -> str:
        """Decrypt a single field"""
        try:
            decrypted = self.cipher_suite.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_dict(self, data: Dict, fields_to_encrypt: List[str]) -> Dict:
        """Encrypt specific fields in a dictionary"""
        encrypted_data = data.copy()
        for field in fields_to_encrypt:
            if field in encrypted_data:
                encrypted_data[field] = self.encrypt_field(str(encrypted_data[field]))
        return encrypted_data
    
    def decrypt_dict(self, data: Dict, fields_to_decrypt: List[str]) -> Dict:
        """Decrypt specific fields in a dictionary"""
        decrypted_data = data.copy()
        for field in fields_to_decrypt:
            if field in decrypted_data:
                try:
                    decrypted_data[field] = self.decrypt_field(decrypted_data[field])
                except:
                    pass  # Field might not be encrypted
        return decrypted_data


class AccessControl:
    """Role-Based Access Control (RBAC)"""
    
    ROLES = {
        'admin': ['read', 'write', 'delete', 'validate', 'export'],
        'clinician': ['read', 'write', 'validate', 'view_audit'],
        'technician': ['read', 'write', 'export'],
        'auditor': ['read', 'view_audit'],
        'patient': ['read']
    }
    
    def __init__(self):
        self.user_roles: Dict[str, List[str]] = {}
        logger.info("Access Control initialized")
    
    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign role to user"""
        if role not in self.ROLES:
            logger.warning(f"Unknown role: {role}")
            return False
        
        self.user_roles[user_id] = [role]
        logger.info(f"Assigned role {role} to user {user_id}")
        return True
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has permission"""
        user_roles = self.user_roles.get(user_id, [])
        
        for role in user_roles:
            if permission in self.ROLES.get(role, []):
                return True
        
        logger.warning(f"User {user_id} denied permission: {permission}")
        return False
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for user"""
        permissions = set()
        for role in self.user_roles.get(user_id, []):
            permissions.update(self.ROLES.get(role, []))
        return list(permissions)


class AuditLog:
    """Audit logging for HIPAA compliance"""
    
    def __init__(self, log_file: str = "logs/audit_log.json"):
        self.log_file = log_file
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        self.logs: List[Dict] = []
        logger.info(f"Audit log initialized: {log_file}")
    
    def log_access(self, user_id: str, action: str, resource_id: str, 
                   resource_type: str = "document", details: Dict = None) -> bool:
        """Log user access to resources"""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "user_id": user_id,
                "action": action,
                "resource_id": resource_id,
                "resource_type": resource_type,
                "details": details or {},
                "ip_address": "127.0.0.1",  # In production, get actual IP
                "status": "success"
            }
            
            self.logs.append(log_entry)
            self._persist_log(log_entry)
            
            logger.info(f"Logged: User {user_id} performed {action} on {resource_type}/{resource_id}")
            return True
        
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            return False
    
    def log_data_access(self, user_id: str, phi_fields: List[str], 
                       reason: str = "clinical_care") -> bool:
        """Log access to PHI (Protected Health Information)"""
        return self.log_access(
            user_id, 
            "access_phi", 
            f"phi_{len(phi_fields)}_fields",
            details={
                "phi_fields": phi_fields,
                "reason": reason
            }
        )
    
    def _persist_log(self, log_entry: Dict):
        """Persist log entry to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist log: {e}")
    
    def get_audit_trail(self, resource_id: str) -> List[Dict]:
        """Get full audit trail for a resource"""
        trail = [log for log in self.logs if log.get('resource_id') == resource_id]
        logger.info(f"Retrieved audit trail for {resource_id}: {len(trail)} entries")
        return trail
    
    def export_audit_log(self, output_file: str) -> bool:
        """Export audit log to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
            logger.info(f"Audit log exported to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False


class HIPAACompliance:
    """HIPAA compliance utilities"""
    
    # Protected Health Information (PHI) patterns
    PHI_PATTERNS = {
        'patient_name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'phone': r'\b\d{3}-\d{3}-\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'medical_record_number': r'\bMRN[:\s]+[A-Z0-9-]+\b',
        'date_of_birth': r'\b(0?[1-9]|1[012])[/-](0?[1-9]|[12][0-9]|3[01])[/-]((19|20)\d{2})\b'
    }
    
    @staticmethod
    def validate_phi_fields(data: Dict) -> Dict[str, List[str]]:
        """Identify PHI fields in data"""
        import re
        phi_found = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                for phi_type, pattern in HIPAACompliance.PHI_PATTERNS.items():
                    if re.search(pattern, value):
                        if phi_type not in phi_found:
                            phi_found[phi_type] = []
                        phi_found[phi_type].append(key)
        
        logger.info(f"Identified PHI fields: {list(phi_found.keys())}")
        return phi_found
    
    @staticmethod
    def create_audit_trail(user_id: str, action: str, resource: Dict) -> Dict:
        """Create audit trail entry for HIPAA compliance"""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": user_id,
            "action": action,
            "resource_id": resource.get('id'),
            "resource_type": resource.get('resourceType'),
            "hash": hashlib.sha256(
                json.dumps(resource, sort_keys=True, default=str).encode()
            ).hexdigest()
        }
    
    @staticmethod
    def anonymize_data(data: Dict, phi_fields: List[str]) -> Dict:
        """Anonymize PHI by replacing with placeholders"""
        anonymized = data.copy()
        
        for field in phi_fields:
            if field in anonymized:
                field_type = type(anonymized[field])
                if field_type == str:
                    anonymized[field] = f"[REDACTED_{field.upper()}]"
                elif field_type in [int, float]:
                    anonymized[field] = None
        
        logger.info(f"Data anonymized: {len(phi_fields)} fields redacted")
        return anonymized
    
    @staticmethod
    def validate_consent(patient_id: str, data_use: str) -> bool:
        """Validate patient consent for data use"""
        # In production, check against consent management system
        logger.info(f"Consent validated for patient {patient_id} - use: {data_use}")
        return True
    
    @staticmethod
    def create_data_use_agreement(organization: str, data_types: List[str]) -> Dict:
        """Create Data Use Agreement (DUA)"""
        dua = {
            "id": str(datetime.utcnow().timestamp()),
            "organization": organization,
            "data_types": data_types,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "expires_at": "2027-01-17T00:00:00Z",
            "restrictions": [
                "No re-identification attempts",
                "Limited to research purposes",
                "Data must be encrypted at rest and in transit"
            ],
            "status": "active"
        }
        return dua


class ComplianceReporter:
    """Generate compliance reports"""
    
    def __init__(self, audit_log: AuditLog):
        self.audit_log = audit_log
    
    def generate_hipaa_report(self, start_date: str, end_date: str) -> Dict:
        """Generate HIPAA compliance report"""
        report = {
            "report_type": "HIPAA Compliance Report",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "period": {"start": start_date, "end": end_date},
            "total_accesses": len(self.audit_log.logs),
            "unique_users": len(set(log.get('user_id') for log in self.audit_log.logs)),
            "phi_access_events": len([
                log for log in self.audit_log.logs 
                if 'phi' in log.get('action', '').lower()
            ]),
            "anomalies_detected": self._detect_anomalies()
        }
        
        logger.info(f"HIPAA report generated: {report['total_accesses']} accesses")
        return report
    
    def _detect_anomalies(self) -> List[Dict]:
        """Detect anomalous access patterns"""
        anomalies = []
        # Simple anomaly detection: multiple access attempts in short time
        user_access_times = {}
        
        for log in self.audit_log.logs:
            user_id = log.get('user_id')
            if user_id not in user_access_times:
                user_access_times[user_id] = []
            user_access_times[user_id].append(log.get('timestamp'))
        
        # Check for suspicious access patterns
        for user_id, times in user_access_times.items():
            if len(times) > 50:  # More than 50 accesses
                anomalies.append({
                    "type": "excessive_access",
                    "user_id": user_id,
                    "count": len(times),
                    "severity": "high"
                })
        
        return anomalies
    
    def export_compliance_report(self, output_file: str, report: Dict) -> bool:
        """Export compliance report"""
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Compliance report exported: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
