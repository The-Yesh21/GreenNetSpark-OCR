import re
import logging
import numpy as np
import cv2
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class VegetableRecognizer:
    """
    Handles Kannada text recognition for vegetable names using PP-OCRv3.
    """
    def __init__(self, use_gpu: bool = False, rec_batch_num: int = 10):
        """
        Initializes the PaddleOCR reader for Kannada language with PP-OCRv3 weights.
        """
        logger.info("Initializing PaddleOCR Kannada reader for vegetables (PP-OCRv3)...")
        device = 'gpu' if use_gpu else 'cpu'
        self.reader = PaddleOCR(
            lang='ka',
            device=device,
            ocr_version='PP-OCRv3',
            rec_batch_num=rec_batch_num
        )

    def recognize(self, crop: np.ndarray) -> tuple:
        """
        Runs OCR on the vegetable crop and returns recognized Kannada text and confidence score.
        
        :param crop: Cropped vegetable bounding box region.
        :return: Tuple of (recognized_text, confidence_score)
        """
        if crop.size == 0:
            return "", 0.0
        try:
            predictor = self.reader.paddlex_pipeline._pipeline.text_rec_model
            results = list(predictor.predict(crop))
            if results:
                res = results[0]
                text = ""
                score = 0.0
                if isinstance(res, dict):
                    text = res.get('rec_text', '').strip()
                    score = float(res.get('rec_score', 0.0))
                elif hasattr(res, 'rec_text'):
                    text = getattr(res, 'rec_text', '').strip()
                    score = float(getattr(res, 'rec_score', 0.0))
                return text, score
            return "", 0.0
        except Exception as e:
            logger.error(f"Error during Kannada vegetable recognition: {e}")
            return "", 0.0


class PriceRecognizer:
    """
    Handles numeric price extraction using PP-OCRv4.
    """
    def __init__(self, use_gpu: bool = False, rec_batch_num: int = 10):
        """
        Initializes the PaddleOCR reader configured for English/Digits with PP-OCRv4 weights.
        """
        logger.info("Initializing PaddleOCR English reader for prices (PP-OCRv4)...")
        device = 'gpu' if use_gpu else 'cpu'
        self.reader = PaddleOCR(
            lang='en',
            device=device,
            ocr_version='PP-OCRv4',
            rec_batch_num=rec_batch_num
        )
        self.regex = re.compile(r'[0-9]+')

    def recognize_and_extract(self, crop: np.ndarray) -> tuple:
        """
        Runs OCR on the price crop and extracts only the first Arabic digit sequence.
        Defaults to '0' if no digits are found.
        Enforces zero character translation logic for Kannada digits by matching
        strictly against ASCII Arabic numerals (0-9).
        
        :param crop: Cropped price bounding box region.
        :return: Tuple of (extracted_digits_text, raw_text, confidence_score)
        """
        if crop.size == 0:
            return "0", "", 0.0
        try:
            predictor = self.reader.paddlex_pipeline._pipeline.text_rec_model
            results = list(predictor.predict(crop))
            text = ""
            score = 0.0
            if results:
                res = results[0]
                if isinstance(res, dict):
                    text = res.get('rec_text', '')
                    score = float(res.get('rec_score', 0.0))
                elif hasattr(res, 'rec_text'):
                    text = getattr(res, 'rec_text', '')
                    score = float(getattr(res, 'rec_score', 0.0))
            
            # Extractor using strictly ASCII [0-9]+ to grab only Arabic numerals (0-9)
            # This implements the no-translate rule, ignoring Kannada Unicode digits.
            digits = re.findall(r'[0-9]+', str(text))
            if digits:
                return "".join(digits), text, score
            return "0", text, score
        except Exception as e:
            logger.error(f"Error during price extraction: {e}")
            return "0", "", 0.0
