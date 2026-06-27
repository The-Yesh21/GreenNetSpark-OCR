import os
import json
import difflib
import logging

logger = logging.getLogger(__name__)

# List of valid, standard Kannada vegetable names for fuzzy matching
VALID_KANNADA_VEGETABLES = [
    "ಆಲೂಗಡ್ಡೆ",    # Potato
    "ಈರುಳ್ಳಿ",      # Onion
    "ಟೊಮೆಟೊ",     # Tomato
    "ಬೆಂಡೆಕಾಯಿ",    # Ladies finger
    "ಬದನೆಕಾಯಿ",    # Brinjal
    "ಕ್ಯಾರೆಟ್",      # Carrot
    "ಮೆಣಸಿನಕಾಯಿ",   # Chilli
    "ಕೊತ್ತಂಬರಿ",    # Coriander
    "ಶುಂಠಿ",        # Ginger
    "ಬೆಳ್ಳುಳ್ಳಿ",     # Garlic
    "ಎಲೆಕೋಸು",     # Cabbage
    "ಹೂಕೋಸು",     # Cauliflower
    "ಸೌತೆಕಾಯಿ",    # Cucumber
    "ಮೂಲಂಗಿ",      # Radish
    "ಪಾಲಕ್",       # Spinach
    "ಗಜ್ಜರಿ",       # Carrot (alternative)
    "ಕುಂಬಳಕಾಯಿ",   # Pumpkin
    "ಮೆಂತೆ ಸೊಪ್ಪು",  # Fenugreek Leaves
    "ಸಬ್ಬಕ್ಕಿ ಸೊಪ್ಪು", # Dill Leaves
    "ಪಡುವಲಕಾಯಿ",   # Snake gourd
    "ಹಾಗಲಕಾಯಿ",    # Bitter gourd
    "ಹೀರೆಕಾಯಿ",     # Ridge gourd
    "ತೊಂಡೆಕಾಯಿ",    # Ivy gourd
    "ನುಗ್ಗೆಕಾಯಿ",    # Drumstick
    "ಸುವರ್ಣಗಡ್ಡೆ",   # Yam
    "ನವಿಲುಕೋಸು",    # Kohlrabi
    "ಕರೇಬೇವು",      # Curry leaves
    "ದಪ್ಪ ಮೆಣಸಿನಕಾಯಿ" # Capsicum
]

# Direct dictionary for common OCR errors or short forms
KANNADA_VEG_MAPPING = {
    "ಇರುಳ್ಳಿ": "ಈರುಳ್ಳಿ",
    "ಇರುಳಿ": "ಈರುಳ್ಳಿ",
    "ಆಲೂಗಡೆ": "ಆಲೂಗಡ್ಡೆ",
    "ಬೆಂಡೆಕಾಯ": "ಬೆಂಡೆಕಾಯಿ",
    "ಬದನೆಕಾಯ": "ಬದನೆಕಾಯಿ",
    "ಬದನೆ": "ಬದನೆಕಾಯಿ",
    "ಟೊಮೇಟೊ": "ಟೊಮೆಟೊ",
    "ಟೊಮೆಟೊ": "ಟೊಮೆಟೊ",
    "ಕ್ಯಾರೇಟ್": "ಕ್ಯಾರೆಟ್",
    "ಮೆಣಸಿನಕಾಯ": "ಮೆಣಸಿನಕಾಯಿ",
}

# Path to the JSON dictionary
VEG_DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "veg_dictionary.json")

def load_veg_dictionary() -> dict:
    """Reads the Kannada vegetable name mappings from a JSON file."""
    if os.path.exists(VEG_DICT_PATH):
        try:
            with open(VEG_DICT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load vegetable dictionary from {VEG_DICT_PATH}: {e}")
    else:
        logger.warning(f"Vegetable dictionary file not found at {VEG_DICT_PATH}. Using empty dictionary.")
    return {}

def correct_vegetable_name(ocr_text: str, cutoff: float = 0.5) -> str:
    """
    Corrects the OCR-recognized Kannada vegetable name using a loaded JSON dictionary
    and fuzzy matching against a list of valid names.
    
    :param ocr_text: Raw OCR recognized text.
    :param cutoff: Fuzzy match similarity threshold (between 0.0 and 1.0).
    :return: Cleaned and corrected Kannada vegetable name.
    """
    if not ocr_text:
        return ""
        
    cleaned_text = ocr_text.strip()
    
    # 1. Load dictionary from JSON and perform 'Best-Match' lookup
    veg_dict = load_veg_dictionary()
    
    best_key = None
    best_score = 0.0
    
    for key in veg_dict.keys():
        score = difflib.SequenceMatcher(None, cleaned_text, key).ratio()
        if score > best_score:
            best_score = score
            best_key = key
            
    if best_score > 0.6 and best_key is not None:
        corrected = veg_dict[best_key]
        logger.info(f"Dictionary Best-Match lookup: '{cleaned_text}' matched '{best_key}' (score={best_score:.2f} > 0.6) -> '{corrected}'")
        return corrected
        
    # 2. Check exact/direct local dictionary mapping fallback
    if cleaned_text in KANNADA_VEG_MAPPING:
        corrected = KANNADA_VEG_MAPPING[cleaned_text]
        logger.debug(f"Direct mapping match fallback: '{cleaned_text}' -> '{corrected}'")
        return corrected
        
    # 3. Use difflib for fuzzy matching fallback
    matches = difflib.get_close_matches(cleaned_text, VALID_KANNADA_VEGETABLES, n=1, cutoff=cutoff)
    if matches:
        corrected = matches[0]
        logger.debug(f"Fuzzy match match (cutoff={cutoff}): '{cleaned_text}' -> '{corrected}'")
        return corrected
        
    logger.debug(f"No match found for: '{cleaned_text}'. Returning original.")
    return cleaned_text
