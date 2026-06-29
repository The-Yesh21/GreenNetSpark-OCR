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

def get_corrected_name(ocr_text: str, cutoff: float = 0.5) -> str:
    """
    Corrects the OCR-recognized Kannada vegetable name.
    If no match is found, returns the raw text with a prefix 'UNCORRECTED_'.
    Also prints input and output directly to the terminal.
    """
    print(f"DEBUG: get_corrected_name Input -> '{ocr_text}'")
    if not ocr_text:
        print("DEBUG: get_corrected_name Output -> ''")
        return ""
        
    cleaned_text = ocr_text.strip()
    
    # Save raw string to ocr_raw_log.txt
    try:
        workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(workspace_dir, "ocr_raw_log.txt")
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"Dictionary Input: '{ocr_text}'\n")
    except Exception as le:
        logger.error(f"Failed to log to ocr_raw_log.txt: {le}")
        
    # 1. Load dictionary from JSON and perform 'Best-Match' lookup using difflib
    veg_dict = load_veg_dictionary()
    
    matches = difflib.get_close_matches(cleaned_text, veg_dict.keys(), n=1, cutoff=0.4)
    if matches:
        corrected = veg_dict[matches[0]]
        print(f"DEBUG: get_corrected_name Output (Dict Fuzzy) -> '{corrected}'")
        return corrected
        
    # 2. Check exact/direct local dictionary mapping fallback
    if cleaned_text in KANNADA_VEG_MAPPING:
        corrected = KANNADA_VEG_MAPPING[cleaned_text]
        print(f"DEBUG: get_corrected_name Output (Direct Map) -> '{corrected}'")
        return corrected
        
    # 3. Use difflib for fuzzy matching fallback against VALID_KANNADA_VEGETABLES
    matches = difflib.get_close_matches(cleaned_text, VALID_KANNADA_VEGETABLES, n=1, cutoff=cutoff)
    if matches:
        corrected = matches[0]
        print(f"DEBUG: get_corrected_name Output (Fuzzy) -> '{corrected}'")
        return corrected
        
    # If no match is found, return raw text prefixed with UNCORRECTED_
    output = f"UNCORRECTED_{cleaned_text}"
    print(f"DEBUG: get_corrected_name Output (Uncorrected) -> '{output}'")
    return output

# Alias for compatibility
correct_vegetable_name = get_corrected_name
