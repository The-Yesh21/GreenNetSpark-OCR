import os
import sys

try:
    import paddleocr
    print("PaddleOCR is successfully installed.")
    print(f"Version: {paddleocr.__version__}")
    print(f"Path: {os.path.dirname(paddleocr.__file__)}")
except ImportError as e:
    print(f"Error importing paddleocr: {e}", file=sys.stderr)
    print("Install dependencies with: python -m pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)
