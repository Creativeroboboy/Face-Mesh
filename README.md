# Egyptian ID Card Reading System

Python/OpenCV OCR pipeline for reading Egyptian ID cards. It detects the card, extracts Arabic text, decodes the Egyptian national ID number, and draws labeled boxes around the name, address, and national-number fields.

## Features

- Egyptian ID card detection and perspective correction
- Arabic and English OCR with EasyOCR and Tesseract
- National ID extraction and decoding
- Date of birth, governorate, and gender extraction
- Face detection and face crop export
- Labeled output image boxes for name, address, and national ID
- JSON report output

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

Read an image:

```powershell
python id_verification_system.py "FAKE ID Card sample.png" --output-dir "outputs\read"
```

Read from webcam:

```powershell
python id_verification_system.py camera --output-dir outputs\camera
```

The main result is saved to:

```text
outputs\read\report.json
```

The annotated card image is saved as:

```text
outputs\read\<image_name>_card_annotated.png
```

## Privacy Note

Do not commit real ID card images, extracted reports, model weights, or generated output files. The `.gitignore` is configured to avoid publishing these by accident.
