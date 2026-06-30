import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytesseract


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
STANDARD_CARD_SIZE = (856, 540)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
EGYPT_GOVERNORATES = {
    "01": "Cairo",
    "02": "Alexandria",
    "03": "Port Said",
    "04": "Suez",
    "11": "Damietta",
    "12": "Dakahlia",
    "13": "Sharqia",
    "14": "Qalyubia",
    "15": "Kafr El Sheikh",
    "16": "Gharbia",
    "17": "Monufia",
    "18": "Beheira",
    "19": "Ismailia",
    "21": "Giza",
    "22": "Beni Suef",
    "23": "Fayoum",
    "24": "Minya",
    "25": "Assiut",
    "26": "Sohag",
    "27": "Qena",
    "28": "Aswan",
    "29": "Luxor",
    "31": "Red Sea",
    "32": "New Valley",
    "33": "Matrouh",
    "34": "North Sinai",
    "35": "South Sinai",
    "88": "Born outside Egypt",
}


@dataclass
class OCRLine:
    text: str
    confidence: float
    box: Tuple[int, int, int, int]


@dataclass
class FaceResult:
    box: Tuple[int, int, int, int]
    confidence: Optional[float] = None
    crop_path: Optional[str] = None


@dataclass
class CardResult:
    detected: bool
    method: str
    corners: Optional[List[List[float]]] = None
    warped_image: Optional[str] = None
    card_crop: Optional[str] = None
    annotated_image: Optional[str] = None
    raw_text: str = ""
    ocr_lines: List[Dict[str, Any]] = field(default_factory=list)
    fields: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ImageReport:
    input: str
    status: str
    output_dir: str
    processed_at: str
    card: CardResult
    faces: List[FaceResult]


class IDCardReadingSystem:
    """Detects, rectifies, reads, and validates ID-card images."""

    def __init__(
        self,
        cascade_path: Optional[str] = None,
        tesseract_cmd: Optional[str] = None,
        yolo_model: Optional[str] = None,
        use_yolo: bool = True,
        profile: str = "egypt",
        languages: str = "eng",
        min_text_confidence: int = 35,
    ) -> None:
        self.root = Path(__file__).resolve().parent
        self.profile = profile
        self.languages = languages
        self.min_text_confidence = min_text_confidence
        self.face_cascade = self._load_face_cascade(cascade_path)
        self.yolo = self._load_yolo(yolo_model) if use_yolo else None
        self.easyocr_reader: Optional[Any] = None
        self.easyocr_failed = False
        self._configure_tesseract(tesseract_cmd)

    def _configure_tesseract(self, tesseract_cmd: Optional[str]) -> None:
        candidates = [
            Path(tesseract_cmd) if tesseract_cmd else None,
            self.root / "Tesseract-OCR" / "tesseract.exe",
            Path("Tesseract-OCR") / "tesseract.exe",
        ]
        for candidate in candidates:
            if candidate and candidate.exists():
                pytesseract.pytesseract.tesseract_cmd = str(candidate)
                return

    def _load_face_cascade(self, cascade_path: Optional[str]) -> Optional[cv2.CascadeClassifier]:
        candidates = [
            Path(cascade_path) if cascade_path else None,
            Path("haarcascade_frontalface_default.xml"),
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades")
            else None,
        ]
        for candidate in candidates:
            if candidate and candidate.exists():
                cascade = cv2.CascadeClassifier(str(candidate))
                if not cascade.empty():
                    return cascade
        return None

    def _load_yolo(self, model_path: Optional[str]) -> Optional[Any]:
        candidates = [
            Path(model_path) if model_path else None,
            self.root / "runs" / "detect" / "my_custom_yolo" / "run_v1-5" / "weights" / "best.pt",
        ]
        model_file = next((path for path in candidates if path and path.exists()), None)
        if model_file is None:
            return None
        try:
            from ultralytics import YOLO

            return YOLO(str(model_file))
        except Exception:
            return None

    def process_path(self, input_path: Path, output_dir: Path) -> Dict[str, Any]:
        if input_path.is_dir():
            return self.process_directory(input_path, output_dir)
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            return self.process_video(input_path, output_dir)
        return asdict(self.process_image(input_path, output_dir))

    def process_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        reports = [asdict(self.process_image(path, output_dir / path.stem)) for path in image_paths]
        csv_path = output_dir / "summary.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_summary_csv(reports, csv_path)
        return {
            "input": str(input_dir),
            "status": "complete",
            "images_processed": len(reports),
            "summary_csv": str(csv_path),
            "reports": reports,
        }

    def process_image(self, input_path: Path, output_dir: Path) -> ImageReport:
        output_dir.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(input_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        card_points, method = self.detect_id_card(image)
        working_image = image
        card_crop_path = None
        warped_path = None
        warnings: List[str] = []

        if card_points is not None and method == "full_frame":
            working_image = image
            if working_image.shape[0] > working_image.shape[1]:
                working_image = cv2.rotate(working_image, cv2.ROTATE_90_CLOCKWISE)
            warped_path = output_dir / f"{input_path.stem}_card_warped.png"
            cv2.imwrite(str(warped_path), working_image)
        elif card_points is not None:
            working_image = self.warp_id_card(image, card_points)
            warped_path = output_dir / f"{input_path.stem}_card_warped.png"
            cv2.imwrite(str(warped_path), working_image)
        else:
            warnings.append("No ID-card boundary was found; OCR was run on the full image.")

        card_crop_path = output_dir / f"{input_path.stem}_ocr_input.png"
        cv2.imwrite(str(card_crop_path), working_image)

        faces = self.detect_faces(working_image)
        face_results = self._save_face_crops(working_image, faces, output_dir, input_path.stem)
        ocr_lines = self.read_text(working_image)
        raw_text = "\n".join(line.text for line in ocr_lines)
        fields = self.extract_fields(raw_text)
        if self.profile == "egypt":
            egyptian_fields = self.extract_egyptian_fields(raw_text, working_image)
            egyptian_fields["field_boxes"] = self.find_egyptian_field_boxes(working_image, ocr_lines, egyptian_fields)
            fields["document_type"] = "egyptian_national_id"
            fields["egyptian_national_id"] = egyptian_fields
            if egyptian_fields.get("national_id"):
                fields["id_number"] = egyptian_fields["national_id"]
                fields["date_of_birth"] = egyptian_fields.get("date_of_birth")
                fields["sex"] = egyptian_fields.get("gender")
        quality = self.assess_quality(working_image, ocr_lines, face_results)
        warnings.extend(self._build_warnings(fields, quality, face_results, ocr_lines))

        annotated = self.annotate(image, card_points, method, faces if card_points is None else [])
        annotated_path = output_dir / f"{input_path.stem}_annotated.png"
        cv2.imwrite(str(annotated_path), annotated)

        if card_points is not None:
            card_annotated = self.annotate_card(working_image, ocr_lines, face_results, fields)
            cv2.imwrite(str(output_dir / f"{input_path.stem}_card_annotated.png"), card_annotated)

        card = CardResult(
            detected=card_points is not None,
            method=method,
            corners=None if card_points is None else card_points.tolist(),
            warped_image=None if warped_path is None else str(warped_path),
            card_crop=str(card_crop_path),
            annotated_image=str(annotated_path),
            raw_text=raw_text,
            ocr_lines=[asdict(line) for line in ocr_lines],
            fields=fields,
            quality=quality,
            warnings=warnings,
        )

        report = ImageReport(
            input=str(input_path),
            status="complete",
            output_dir=str(output_dir),
            processed_at=datetime.now().isoformat(timespec="seconds"),
            card=card,
            faces=face_results,
        )
        self._write_json(asdict(report), output_dir / f"{input_path.stem}_report.json")
        return report

    def process_video(self, input_path: Path, output_dir: Path, sample_every: int = 30) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_video = output_dir / f"{input_path.stem}_processed.mp4"
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        best_frame: Optional[np.ndarray] = None
        best_score = -1
        frame_count = 0
        cards_detected = 0
        faces_total = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            card_points, method = self.detect_id_card(frame)
            faces = self.detect_faces(frame)
            annotated = self.annotate(frame, card_points, method, faces)
            writer.write(annotated)

            score = (100 if card_points is not None else 0) + len(faces) * 10
            if frame_count % sample_every == 0 and score > best_score:
                best_frame = frame.copy()
                best_score = score
            frame_count += 1
            cards_detected += int(card_points is not None)
            faces_total += len(faces)

        cap.release()
        writer.release()

        still_report = None
        if best_frame is not None:
            best_frame_path = output_dir / f"{input_path.stem}_best_frame.png"
            cv2.imwrite(str(best_frame_path), best_frame)
            still_report = asdict(self.process_image(best_frame_path, output_dir / "best_frame_read"))

        return {
            "input": str(input_path),
            "status": "complete",
            "output_video": str(output_video),
            "frames_processed": frame_count,
            "card_detections": cards_detected,
            "faces_detected": faces_total,
            "best_frame_report": still_report,
        }

    def process_live_camera(self, output_dir: Path, camera_index: int = 0, display: bool = True) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to access webcam index {camera_index}.")

        best_frame: Optional[np.ndarray] = None
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            card_points, method = self.detect_id_card(frame)
            faces = self.detect_faces(frame)
            annotated = self.annotate(frame, card_points, method, faces)
            if card_points is not None and faces:
                best_frame = frame.copy()
            elif best_frame is None and card_points is not None:
                best_frame = frame.copy()

            frame_count += 1
            if display:
                cv2.imshow("ID Card Reading System - press q to finish", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if display:
            cv2.destroyAllWindows()

        report = {"status": "complete", "frames_processed": frame_count, "best_frame_report": None}
        if best_frame is not None:
            frame_path = output_dir / "camera_capture.png"
            cv2.imwrite(str(frame_path), best_frame)
            report["best_frame_report"] = asdict(self.process_image(frame_path, output_dir / "camera_read"))
        return report

    def detect_id_card(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        yolo_points = self._detect_id_card_yolo(image)
        if yolo_points is not None:
            return yolo_points, "yolo"

        contour_points = self._detect_id_card_contours(image)
        if contour_points is not None:
            return contour_points, "contour"

        full_frame_points = self._detect_full_frame_card(image)
        if full_frame_points is not None:
            return full_frame_points, "full_frame"

        return None, "none"

    def _detect_id_card_yolo(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.yolo is None:
            return None
        try:
            results = self.yolo.predict(source=image, verbose=False, conf=0.25)
        except Exception:
            return None
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    def _detect_id_card_contours(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_area = image.shape[0] * image.shape[1]
        candidates = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            area = cv2.contourArea(contour)
            if area < image_area * 0.05:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                points = self._order_points(approx.reshape(4, 2).astype(np.float32))
                ratio = self._side_ratio(points)
                score = area - abs(ratio - 1.586) * 10000
                candidates.append((score, points))

        if candidates:
            return max(candidates, key=lambda item: item[0])[1]

        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if w * h > image_area * 0.12:
            return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        return None

    def _detect_full_frame_card(self, image: np.ndarray) -> Optional[np.ndarray]:
        height, width = image.shape[:2]
        ratio = width / max(height, 1)
        landscape_like = 1.2 <= ratio <= 1.9
        portrait_like = 0.52 <= ratio <= 0.85
        if not (landscape_like or portrait_like):
            return None
        return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.face_cascade is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(40, 40),
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def read_text(self, image: np.ndarray) -> List[OCRLine]:
        prepared_images = self._ocr_variants(image)
        best_lines: List[OCRLine] = []
        for prepared in prepared_images:
            lines = self._run_tesseract(prepared)
            if self._ocr_score(lines) > self._ocr_score(best_lines):
                best_lines = lines
        if self.profile == "egypt":
            best_lines = self._merge_ocr_lines(best_lines, self._run_easyocr(image))
        return best_lines

    def _run_tesseract(self, image: np.ndarray) -> List[OCRLine]:
        config = "--oem 3 --psm 6"
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.languages,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            return []

        grouped: Dict[Tuple[int, int, int], List[int]] = {}
        for index, text in enumerate(data.get("text", [])):
            clean = " ".join(text.split())
            if not clean:
                continue
            try:
                conf = float(data["conf"][index])
            except (ValueError, TypeError):
                conf = -1
            if conf < self.min_text_confidence:
                continue
            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            grouped.setdefault(key, []).append(index)

        lines: List[OCRLine] = []
        for indexes in grouped.values():
            words = [" ".join(data["text"][i].split()) for i in indexes]
            confs = [float(data["conf"][i]) for i in indexes if float(data["conf"][i]) >= 0]
            left = min(data["left"][i] for i in indexes)
            top = min(data["top"][i] for i in indexes)
            right = max(data["left"][i] + data["width"][i] for i in indexes)
            bottom = max(data["top"][i] + data["height"][i] for i in indexes)
            text = " ".join(word for word in words if word)
            if text:
                lines.append(OCRLine(text=text, confidence=round(float(np.mean(confs)) if confs else 0, 2), box=(left, top, right - left, bottom - top)))

        return sorted(lines, key=lambda line: (line.box[1], line.box[0]))

    def _get_easyocr_reader(self) -> Optional[Any]:
        if self.easyocr_failed:
            return None
        if self.easyocr_reader is not None:
            return self.easyocr_reader
        try:
            import easyocr

            self.easyocr_reader = easyocr.Reader(["ar", "en"], gpu=False, verbose=False)
        except Exception:
            self.easyocr_failed = True
            return None
        return self.easyocr_reader

    def _run_easyocr(self, image: np.ndarray, offset: Tuple[int, int] = (0, 0), allowlist: Optional[str] = None) -> List[OCRLine]:
        reader = self._get_easyocr_reader()
        if reader is None:
            return []
        try:
            results = reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist=allowlist,
            )
        except Exception:
            return []

        lines = []
        ox, oy = offset
        for bbox, text, confidence in results:
            clean = " ".join(str(text).split())
            if not clean:
                continue
            points = np.array(bbox, dtype=np.float32)
            x1 = int(np.min(points[:, 0])) + ox
            y1 = int(np.min(points[:, 1])) + oy
            x2 = int(np.max(points[:, 0])) + ox
            y2 = int(np.max(points[:, 1])) + oy
            lines.append(OCRLine(text=clean, confidence=round(float(confidence) * 100, 2), box=(x1, y1, x2 - x1, y2 - y1)))
        return sorted(lines, key=lambda line: (line.box[1], line.box[0]))

    def _merge_ocr_lines(self, primary: Sequence[OCRLine], secondary: Sequence[OCRLine]) -> List[OCRLine]:
        merged = list(primary)
        seen = {self._normalize_for_matching(line.text) for line in merged}
        for line in secondary:
            key = self._normalize_for_matching(line.text)
            if key and key not in seen:
                seen.add(key)
                merged.append(line)
        return sorted(merged, key=lambda line: (line.box[1], line.box[0], line.text))

    def extract_fields(self, raw_text: str) -> Dict[str, Any]:
        text = self._normalize_text(raw_text)
        lines = [line.strip(" :|-") for line in text.splitlines() if line.strip(" :|-")]
        joined = "\n".join(lines)

        fields: Dict[str, Any] = {
            "document_numbers": self._unique(re.findall(r"\b[A-Z0-9]{6,14}\b", joined)),
            "dates": self._unique(self._extract_dates(joined)),
            "mrz": self._extract_mrz(lines),
        }

        label_patterns = {
            "name": r"(?:NAME|FULL NAME|NAMES?)[:\s-]+([A-Z][A-Z\s.'-]{2,})",
            "surname": r"(?:SURNAME|LAST NAME)[:\s-]+([A-Z][A-Z\s.'-]{2,})",
            "nationality": r"(?:NATIONALITY|NATION)[:\s-]+([A-Z\s]{3,})",
            "sex": r"(?:SEX|GENDER)[:\s-]+([MF]|MALE|FEMALE)\b",
            "date_of_birth": r"(?:DOB|DATE OF BIRTH|BIRTH)[:\s-]+([0-9]{1,4}[./\-\s][0-9A-Z]{1,3}[./\-\s][0-9]{2,4})",
            "expiry_date": r"(?:EXPIRY|EXPIRES|VALID UNTIL|EXP)[:\s-]+([0-9]{1,4}[./\-\s][0-9A-Z]{1,3}[./\-\s][0-9]{2,4})",
            "id_number": r"(?:ID|IDENTITY|DOCUMENT|CARD|LICENSE|LICENCE|NO|NUMBER)[:#\s-]+([A-Z0-9-]{5,20})",
        }
        for key, pattern in label_patterns.items():
            match = re.search(pattern, joined, flags=re.IGNORECASE)
            if match:
                fields[key] = match.group(1).strip()

        if "name" not in fields:
            fields["possible_name"] = self._guess_name(lines)
        if fields["mrz"]:
            fields["mrz_fields"] = self._parse_mrz(fields["mrz"])

        fields["raw_lines"] = lines
        return fields

    def extract_egyptian_fields(self, raw_text: str, image: np.ndarray) -> Dict[str, Any]:
        candidates = self._extract_egyptian_id_candidates(raw_text)
        region_texts = self._read_egyptian_number_regions(image)
        for text in region_texts:
            candidates.extend(self._extract_egyptian_id_candidates(text))
        candidates.extend(self._combine_egyptian_digit_fragments(region_texts))

        best_number = self._choose_best_egyptian_id(candidates)
        decoded = self._decode_egyptian_national_id(best_number) if best_number else {}
        arabic_lines = self._extract_arabic_lines(raw_text)

        return {
            "national_id": best_number,
            "valid_national_id": bool(decoded),
            "date_of_birth": decoded.get("date_of_birth"),
            "birth_century": decoded.get("birth_century"),
            "governorate_code": decoded.get("governorate_code"),
            "governorate": decoded.get("governorate"),
            "gender": decoded.get("gender"),
            "check_digit": decoded.get("check_digit"),
            "candidate_numbers": self._unique(candidates),
            "unverified_national_id_candidate": self._unique(candidates)[0] if candidates and not best_number else None,
            "number_region_text": region_texts,
            "arabic_name_lines": self._guess_egyptian_name_lines(arabic_lines),
            "arabic_address_lines": self._guess_egyptian_address_lines(arabic_lines),
        }

    def find_egyptian_field_boxes(
        self,
        image: np.ndarray,
        ocr_lines: Sequence[OCRLine],
        egyptian_fields: Dict[str, Any],
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        height, width = image.shape[:2]
        name_lines = egyptian_fields.get("arabic_name_lines", [])
        address_lines = egyptian_fields.get("arabic_address_lines", [])
        national_id = egyptian_fields.get("national_id")

        name_box = self._box_for_text_lines(ocr_lines, name_lines, image.shape)
        address_box = self._box_for_text_lines(ocr_lines, address_lines, image.shape)
        number_box = self._box_for_national_id(ocr_lines, national_id, image.shape)

        if number_box is None:
            number_box = self._relative_box(width, height, 0.40, 0.66, 0.97, 0.86)

        return {
            "name": self._field_box("name", name_lines, name_box),
            "address": self._field_box("address", address_lines, address_box),
            "national_id": self._field_box("national_id", national_id, number_box),
        }

    def _box_for_text_lines(
        self,
        ocr_lines: Sequence[OCRLine],
        target_lines: Sequence[str],
        image_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, int, int, int]]:
        target_keys = {self._normalize_for_matching(line) for line in target_lines if line}
        if not target_keys:
            return None

        matched_boxes = []
        for line in ocr_lines:
            key = self._normalize_for_matching(line.text)
            if not key:
                continue
            if key in target_keys or any(key in target or target in key for target in target_keys):
                matched_boxes.append(line.box)
        return self._union_boxes(matched_boxes, image_shape, padding=12)

    def _box_for_national_id(
        self,
        ocr_lines: Sequence[OCRLine],
        national_id: Optional[str],
        image_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, int, int, int]]:
        if not national_id:
            return None
        height = image_shape[0]
        matched_boxes = []
        for line in ocr_lines:
            digits = re.sub(r"\D+", "", self._normalize_digits(line.text))
            if not digits:
                continue
            x, y, w, h = line.box
            if y < height * 0.55:
                continue
            if digits in national_id or national_id in digits:
                matched_boxes.append((x, y, w, h))
        return self._union_boxes(matched_boxes, image_shape, padding=16)

    def _field_box(
        self,
        label: str,
        value: Any,
        box: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Dict[str, Any]]:
        if box is None:
            return None
        return {
            "label": label,
            "value": value,
            "box": {
                "x": int(box[0]),
                "y": int(box[1]),
                "width": int(box[2]),
                "height": int(box[3]),
            },
        }

    def _union_boxes(
        self,
        boxes: Sequence[Tuple[int, int, int, int]],
        image_shape: Tuple[int, ...],
        padding: int = 0,
    ) -> Optional[Tuple[int, int, int, int]]:
        if not boxes:
            return None
        height, width = image_shape[:2]
        x1 = max(0, min(x for x, _, _, _ in boxes) - padding)
        y1 = max(0, min(y for _, y, _, _ in boxes) - padding)
        x2 = min(width - 1, max(x + w for x, _, w, _ in boxes) + padding)
        y2 = min(height - 1, max(y + h for _, y, _, h in boxes) + padding)
        return (x1, y1, x2 - x1, y2 - y1)

    def _relative_box(
        self,
        width: int,
        height: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Tuple[int, int, int, int]:
        left = int(width * x1)
        top = int(height * y1)
        right = int(width * x2)
        bottom = int(height * y2)
        return (left, top, right - left, bottom - top)

    def _read_egyptian_number_regions(self, image: np.ndarray) -> List[str]:
        height, width = image.shape[:2]
        regions = [
            (int(width * 0.36), int(height * 0.62), int(width * 0.98), int(height * 0.90)),
            (int(width * 0.42), int(height * 0.68), int(width * 0.98), int(height * 0.86)),
            (0, int(height * 0.62), width, int(height * 0.92)),
        ]
        allowlist = "0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
        texts: List[str] = []
        for x1, y1, x2, y2 in regions:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            variants = self._digit_ocr_variants(crop)
            for variant in variants:
                lines = self._run_easyocr(variant, allowlist=allowlist)
                texts.extend(line.text for line in lines)
        return self._unique(texts)

    def _digit_ocr_variants(self, image: np.ndarray) -> List[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.fastNlMeansDenoising(scaled, h=9)
        contrast = cv2.convertScaleAbs(denoised, alpha=1.7, beta=-20)
        adaptive = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
        return [image, scaled, denoised, contrast, adaptive]

    def _extract_egyptian_id_candidates(self, text: str) -> List[str]:
        normalized = self._normalize_digits(text)
        compact = re.sub(r"\D+", "", normalized)
        candidates = re.findall(r"[23]\d{13}", compact)

        spaced_candidates = []
        for match in re.finditer(r"(?:[23][\d\s.\-/]{13,40})", normalized):
            digits = re.sub(r"\D+", "", match.group(0))
            if len(digits) >= 14:
                spaced_candidates.extend(re.findall(r"[23]\d{13}", digits))

        return self._unique(candidates + spaced_candidates)

    def _choose_best_egyptian_id(self, candidates: Sequence[str]) -> Optional[str]:
        valid = [candidate for candidate in candidates if self._decode_egyptian_national_id(candidate)]
        if valid:
            return valid[0]
        return None

    def _combine_egyptian_digit_fragments(self, fragments: Sequence[str]) -> List[str]:
        digit_fragments = []
        for fragment in fragments:
            digits = re.sub(r"\D+", "", self._normalize_digits(fragment))
            if digits:
                digit_fragments.append(digits)

        candidates: List[str] = []
        for start in range(len(digit_fragments)):
            combined = ""
            for fragment in digit_fragments[start : start + 5]:
                combined += fragment
                if len(combined) >= 14:
                    candidates.extend(re.findall(r"[23]\d{13}", combined))
                    break
        return self._unique(candidates)

    def _decode_egyptian_national_id(self, number: Optional[str]) -> Dict[str, Any]:
        if not number or not re.fullmatch(r"[23]\d{13}", number):
            return {}

        century_digit = number[0]
        year = int(number[1:3])
        month = int(number[3:5])
        day = int(number[5:7])
        governorate_code = number[7:9]
        sequence_gender_digit = int(number[12])
        full_year = (1900 if century_digit == "2" else 2000) + year
        governorate = EGYPT_GOVERNORATES.get(governorate_code)
        if governorate is None:
            return {}

        try:
            birth_date = datetime(full_year, month, day).date()
        except ValueError:
            return {}

        return {
            "date_of_birth": birth_date.isoformat(),
            "birth_century": "1900s" if century_digit == "2" else "2000s",
            "governorate_code": governorate_code,
            "governorate": governorate,
            "gender": "Male" if sequence_gender_digit % 2 else "Female",
            "check_digit": number[-1],
        }

    def _extract_arabic_lines(self, text: str) -> List[str]:
        lines = []
        for line in text.splitlines():
            clean = " ".join(line.split())
            if re.search(r"[\u0600-\u06FF]", clean):
                lines.append(clean)
        return lines

    def _guess_egyptian_name_lines(self, arabic_lines: Sequence[str]) -> List[str]:
        blocked = ("جمهورية", "جمهوز", "بطاقة", "الشخصية", "تحقيق", "العربية")
        address_markers = ("شارع", "برج", "طنطا", "القاهرة", "الجيزة", "الاسكندرية", "الغربية", "محافظة", "قسم", "مركز")
        candidates = []
        for line in arabic_lines:
            if any(word in line for word in blocked):
                continue
            if any(marker in line for marker in address_markers):
                continue
            if re.search(r"\d", self._normalize_digits(line)):
                continue
            if 1 <= len(line.split()) <= 5:
                candidates.append(line)
        return candidates[:3]

    def _guess_egyptian_address_lines(self, arabic_lines: Sequence[str]) -> List[str]:
        address_markers = ("شارع", "برج", "طنطا", "القاهرة", "الجيزة", "الاسكندرية", "الغربية", "محافظة", "قسم", "مركز")
        return [line for line in arabic_lines if any(marker in line for marker in address_markers)]

    def assess_quality(self, image: np.ndarray, ocr_lines: Sequence[OCRLine], faces: Sequence[FaceResult]) -> Dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        text_confidence = float(np.mean([line.confidence for line in ocr_lines])) if ocr_lines else 0.0
        return {
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "ocr_line_count": len(ocr_lines),
            "average_ocr_confidence": round(text_confidence, 2),
            "face_count": len(faces),
            "is_blurry": blur_score < 80,
            "is_too_dark": brightness < 55,
            "is_too_bright": brightness > 220,
            "has_readable_text": len(ocr_lines) >= 2 and text_confidence >= self.min_text_confidence,
            "has_face": len(faces) > 0,
        }

    def warp_id_card(self, image: np.ndarray, points: np.ndarray, size: Tuple[int, int] = STANDARD_CARD_SIZE) -> np.ndarray:
        width, height = size
        rect = self._order_points(points)
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (width, height))

    def annotate(
        self,
        image: np.ndarray,
        card_points: Optional[np.ndarray],
        method: str,
        faces: Sequence[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        annotated = image.copy()
        for x, y, w, h in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 180, 0), 2)
            cv2.putText(annotated, "FACE", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
        if card_points is not None:
            pts = card_points.astype(int).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (255, 80, 0), 3)
            x, y = pts.reshape(4, 2)[0]
            cv2.putText(annotated, f"ID CARD ({method})", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 0), 2)
        return annotated

    def annotate_card(
        self,
        image: np.ndarray,
        ocr_lines: Sequence[OCRLine],
        faces: Sequence[FaceResult],
        fields: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        annotated = image.copy()
        for face in faces:
            x, y, w, h = face.box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 180, 0), 2)
        for line in ocr_lines:
            x, y, w, h = line.box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (170, 170, 170), 1)
        self._draw_field_boxes(annotated, fields or {})
        return annotated

    def _draw_field_boxes(self, image: np.ndarray, fields: Dict[str, Any]) -> None:
        egyptian = fields.get("egyptian_national_id", {})
        field_boxes = egyptian.get("field_boxes", {}) if isinstance(egyptian, dict) else {}
        styles = {
            "name": ((40, 120, 255), "NAME"),
            "address": ((255, 80, 0), "ADDRESS"),
            "national_id": ((0, 0, 255), "NATIONAL ID"),
        }
        for key, (color, label) in styles.items():
            item = field_boxes.get(key)
            if not item:
                continue
            box = item.get("box", {})
            x = int(box.get("x", 0))
            y = int(box.get("y", 0))
            w = int(box.get("width", 0))
            h = int(box.get("height", 0))
            if w <= 0 or h <= 0:
                continue
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            label_y = max(22, y - 8)
            cv2.putText(image, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _ocr_variants(self, image: np.ndarray) -> List[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.fastNlMeansDenoising(scaled, h=12)
        adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [scaled, denoised, adaptive, otsu]

    def _save_face_crops(
        self,
        image: np.ndarray,
        faces: Sequence[Tuple[int, int, int, int]],
        output_dir: Path,
        stem: str,
    ) -> List[FaceResult]:
        results: List[FaceResult] = []
        for index, (x, y, w, h) in enumerate(faces, start=1):
            pad_x = int(w * 0.12)
            pad_y = int(h * 0.16)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.shape[1], x + w + pad_x)
            y2 = min(image.shape[0], y + h + pad_y)
            crop_path = output_dir / f"{stem}_face_{index}.png"
            cv2.imwrite(str(crop_path), image[y1:y2, x1:x2])
            results.append(FaceResult(box=(x, y, w, h), crop_path=str(crop_path)))
        return results

    def _build_warnings(
        self,
        fields: Dict[str, Any],
        quality: Dict[str, Any],
        faces: Sequence[FaceResult],
        ocr_lines: Sequence[OCRLine],
    ) -> List[str]:
        warnings = []
        if not faces:
            warnings.append("No face portrait was detected.")
        if not ocr_lines:
            warnings.append("No readable OCR text was detected.")
        if not fields.get("document_numbers") and not fields.get("id_number"):
            warnings.append("No document or ID number was confidently extracted.")
        if quality["is_blurry"]:
            warnings.append("Image appears blurry; use a sharper photo.")
        if quality["is_too_dark"] or quality["is_too_bright"]:
            warnings.append("Lighting is outside the recommended range.")
        return warnings

    def _write_summary_csv(self, reports: Sequence[Dict[str, Any]], csv_path: Path) -> None:
        rows = []
        for report in reports:
            fields = report.get("card", {}).get("fields", {})
            quality = report.get("card", {}).get("quality", {})
            rows.append(
                {
                    "input": report.get("input"),
                    "card_detected": report.get("card", {}).get("detected"),
                    "faces": len(report.get("faces", [])),
                    "name": fields.get("name") or fields.get("possible_name"),
                    "id_number": fields.get("id_number") or ", ".join(fields.get("document_numbers", [])[:2]),
                    "dates": ", ".join(fields.get("dates", [])[:3]),
                    "ocr_lines": quality.get("ocr_line_count"),
                    "ocr_confidence": quality.get("average_ocr_confidence"),
                    "warnings": " | ".join(report.get("card", {}).get("warnings", [])),
                }
            )
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["input"])
            writer.writeheader()
            writer.writerows(rows)

    def _write_json(self, data: Dict[str, Any], path: Path) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        return rect

    def _side_ratio(self, points: np.ndarray) -> float:
        width = np.linalg.norm(points[1] - points[0])
        height = np.linalg.norm(points[3] - points[0])
        if height == 0:
            return 0.0
        return float(width / height)

    def _ocr_score(self, lines: Sequence[OCRLine]) -> float:
        return sum(len(line.text) * max(line.confidence, 0) for line in lines)

    def _normalize_text(self, text: str) -> str:
        text = text.upper()
        text = text.replace("|", "I").replace("€", "E").replace("—", "-")
        return re.sub(r"[ \t]+", " ", text)

    def _normalize_digits(self, text: str) -> str:
        return text.translate(ARABIC_DIGITS)

    def _normalize_for_matching(self, text: str) -> str:
        return re.sub(r"\W+", "", self._normalize_digits(text).casefold())

    def _extract_dates(self, text: str) -> List[str]:
        patterns = [
            r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
            r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b",
            r"\b\d{1,2}\s(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s\d{2,4}\b",
        ]
        dates: List[str] = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text))
        return dates

    def _extract_mrz(self, lines: Sequence[str]) -> List[str]:
        mrz_lines = [line.replace(" ", "") for line in lines if "<" in line and len(line.replace(" ", "")) >= 25]
        return mrz_lines[-3:]

    def _parse_mrz(self, mrz_lines: Sequence[str]) -> Dict[str, Any]:
        if len(mrz_lines) < 2:
            return {}
        joined = "\n".join(mrz_lines)
        result = {"raw": list(mrz_lines)}
        doc_match = re.search(r"([A-Z0-9<]{1,2})([A-Z]{3})([A-Z<]+)", joined)
        if doc_match:
            result["issuing_country"] = doc_match.group(2).replace("<", "")
        number_match = re.search(r"\n([A-Z0-9<]{6,12})", joined)
        if number_match:
            result["document_number"] = number_match.group(1).replace("<", "")
        names = mrz_lines[0].split("<<", 1)
        if len(names) == 2:
            result["surname"] = names[0].split("<")[-1]
            result["given_names"] = names[1].replace("<", " ").strip()
        return result

    def _guess_name(self, lines: Sequence[str]) -> Optional[str]:
        blocked = {"IDENTITY", "CARD", "REPUBLIC", "GOVERNMENT", "NATIONALITY", "DATE", "BIRTH", "EXPIRY"}
        for line in lines:
            words = [word for word in re.findall(r"[A-Z]{2,}", line) if word not in blocked]
            if 2 <= len(words) <= 5 and not any(char.isdigit() for char in line):
                return " ".join(words)
        return None

    def _unique(self, values: Iterable[str]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            clean = value.strip()
            if clean and clean not in seen:
                seen.add(clean)
                result.append(clean)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full ID card reading system")
    parser.add_argument("input", nargs="?", default="face.jpg", help="Image, folder, video, or 0/camera/webcam")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated images and reports")
    parser.add_argument("--cascade", default=None, help="Optional path to a custom OpenCV face cascade XML")
    parser.add_argument("--tesseract", default=None, help="Optional path to tesseract.exe")
    parser.add_argument("--yolo-model", default=None, help="Optional YOLO ID-card detector weights")
    parser.add_argument("--no-yolo", action="store_true", help="Use contour card detection only")
    parser.add_argument("--profile", choices=["egypt", "generic"], default="egypt", help="ID parsing profile")
    parser.add_argument("--languages", default="eng", help="Tesseract language codes, for example eng or eng+ara")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index when input is camera")
    parser.add_argument("--no-display", action="store_true", help="Do not show webcam preview windows")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    system = IDCardReadingSystem(
        cascade_path=args.cascade,
        tesseract_cmd=args.tesseract,
        yolo_model=args.yolo_model,
        use_yolo=not args.no_yolo,
        profile=args.profile,
        languages=args.languages,
    )
    output_dir = Path(args.output_dir)

    if str(args.input).lower() in {"0", "camera", "webcam"}:
        report = system.process_live_camera(output_dir, camera_index=args.camera_index, display=not args.no_display)
    else:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            parser.error(f"Input path does not exist: {input_path}")
        report = system.process_path(input_path, output_dir)

    report_path = output_dir / "report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())