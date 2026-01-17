import logging
import os
import json
import re
from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("pytesseract not installed. Install with: pip install pytesseract")

logger = logging.getLogger(__name__)

class OCRModelWrapper:
    """Wrapper for Tesseract OCR engine using pytesseract"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize OCR wrapper using pytesseract (Tesseract-OCR engine)
        
        Args:
            model_path: Optional path (not used with pytesseract, but kept for compatibility)
        """
        self.model_path = model_path
        self.model = None
        
        if not PYTESSERACT_AVAILABLE:
            raise ImportError("pytesseract is required. Install with: pip install pytesseract")
        
        logger.info("Initialized pytesseract OCR wrapper (Tesseract-OCR engine)")
        logger.info("Note: Requires Tesseract-OCR to be installed on system")
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess image for OCR"""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too small (for better OCR)
            if image.width < 300 or image.height < 100:
                scale_factor = max(300 / image.width, 100 / image.height)
                new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Downsize if too large
            max_size = 3000
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            logger.debug(f"Image preprocessed: {image.size}")
            return image
        
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image_path: Path to medical document image
            
        Returns:
            Dictionary with extracted text and confidence scores
        """
        try:
            # Preprocess image
            image = self.preprocess_image(image_path)
            if image is None:
                return {'text': '', 'confidence': 0.0, 'error': 'Image preprocessing failed'}
            
            # Extract text using pytesseract
            extracted_text = pytesseract.image_to_string(image, lang='eng')
            
            # Get detailed information
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Calculate confidence from Tesseract data
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.5
            
            # Clean up text
            cleaned_text = self._clean_ocr_text(extracted_text)
            
            return {
                'text': cleaned_text,
                'confidence': float(avg_confidence),
                'image_path': image_path,
                'raw_text': extracted_text,
                'word_count': len(cleaned_text.split()),
                'character_count': len(cleaned_text)
            }
        
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return {'text': '', 'confidence': 0.0, 'error': str(e)}
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean and normalize OCR text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common OCR errors
        replacements = {
            'l0': 'lo',  # l zero to lo
            'O0': 'O',   # O zero
            '1l': 'il',  # 1 l to il
            '5S': 'S',   # 5 S to S
        }
        
        for error, correct in replacements.items():
            text = text.replace(error, correct)
        
        return text
    
    def extract_text_with_boxes(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text with bounding boxes and detailed information
        
        Args:
            image_path: Path to medical document image
            
        Returns:
            Dictionary with text, boxes, and metadata
        """
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return {'error': 'Image preprocessing failed'}
            
            # Get detailed box data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            boxes = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:
                    boxes.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]) / 100.0,
                        'bbox': {
                            'left': int(data['left'][i]),
                            'top': int(data['top'][i]),
                            'width': int(data['width'][i]),
                            'height': int(data['height'][i])
                        },
                        'block_num': int(data['block_num'][i]),
                        'page_num': int(data['page_num'][i]),
                        'par_num': int(data['par_num'][i]),
                        'line_num': int(data['line_num'][i]),
                        'word_num': int(data['word_num'][i])
                    })
            
            # Extract full text
            full_text = ' '.join([box['text'] for box in boxes if box['text'].strip()])
            
            return {
                'text': full_text,
                'boxes': boxes,
                'confidence': np.mean([b['confidence'] for b in boxes]) if boxes else 0.0,
                'word_count': len(boxes)
            }
        
        except Exception as e:
            logger.error(f"Extract text with boxes failed: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _parse_predictions(predictions) -> str:
        """Legacy method for compatibility"""
        # Not used with pytesseract, but kept for compatibility
        return str(predictions)
    
    @staticmethod
    def _decode_character_predictions(char_predictions) -> str:
        """Legacy method for compatibility"""
        # Not used with pytesseract, but kept for compatibility
        return str(char_predictions)
    
    @staticmethod
    def _calculate_confidence(predictions) -> float:
        """Legacy method for compatibility"""
        # Not used with pytesseract, but kept for compatibility
        return 0.5
    
    def batch_extract_text(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Extract text from multiple images"""
        results = []
        for image_path in image_paths:
            result = self.extract_text(image_path)
            results.append(result)
        return results
