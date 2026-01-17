import logging
import yaml
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    """Load and manage configuration from YAML file"""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.load_config()
    
    def load_config(self):
        """Load YAML configuration file"""
        # Try multiple paths
        possible_paths = [
            Path(__file__).parent.parent.parent / "config" / "config.yaml",  # Root/config/
            Path(__file__).parent / "config.yaml",  # src/utils/
            Path.cwd() / "config" / "config.yaml",  # Current working directory
        ]
        
        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                self._config = {}
        else:
            logger.warning(f"Config file not found in any of: {possible_paths}. Using defaults.")
            self._config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'ocr.model_path')"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration"""
        return self._config


class LoggerSetup:
    """Configure logging with proper formatting and handlers"""
    
    @staticmethod
    def setup():
        """Setup logging configuration"""
        config = ConfigManager()
        log_file = config.get('logging.file', 'logs/emr_digitization.log')
        log_level = config.get('logging.level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
