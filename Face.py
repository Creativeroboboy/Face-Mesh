import os
import sys
import numpy as np
import cv2


def get_face_cascade_path():
    """Locate the Haar cascade XML using OpenCV's bundled data path."""
    local_candidates = ["haarcascade_frontalface_default.xml"]

    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        local_candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    for candidate in local_candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return None


def create_synthetic_face_canvas():
    """Generates a gray baseline canvas containing two simulated face silhouettes."""
    canvas = np.ones((500, 700, 3), dtype=np.uint8) * 200
    
    # Fake Face 1 (Left Side Structure)
    cv2.circle(canvas, (200, 250), 80, (140, 140, 140), -1)  # Head outline
    cv2.circle(canvas, (170, 230), 10, (40, 40, 40), -1)     # Left Eye
    cv2.circle(canvas, (230, 230), 10, (40, 40, 40), -1)     # Right Eye
    cv2.rectangle(canvas, (180, 280), (220, 290), (40, 40, 40), -1)  # Mouth
    
    # Fake Face 2 (Right Side Structure - Smaller/Scale Test)
    cv2.circle(canvas, (500, 250), 50, (140, 140, 140), -1)  # Head outline
    cv2.circle(canvas, (480, 240), 6, (40, 40, 40), -1)      # Left Eye
    cv2.circle(canvas, (520, 240), 6, (40, 40, 40), -1)      # Right Eye
    cv2.rectangle(canvas, (490, 270), (510, 276), (40, 40, 40), -1)  # Mouth
    
    return canvas

def main():
    cascade_filename = get_face_cascade_path()

    # 1. Verify XML file configuration
    if not cascade_filename:
        print("[ERROR] 'haarcascade_frontalface_default.xml' could not be found.")
        print("Please ensure OpenCV is installed correctly and includes its Haar cascade data.")
        sys.exit(1)

    # 2. Load the Classifier Weights
    face_cascade = cv2.CascadeClassifier(cascade_filename)

    # 3. Read image and transform to Grayscale
    # Haar Cascades require a 1-channel Grayscale array to process features
    image = create_synthetic_face_canvas()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. Execute Face Detection
    # Returns a list of bounding boxes: [[x, y, width, height], ...]
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,    # Shrinks image size by 10% per scale layer evaluation
        minNeighbors=5,     # Requires 5 overlapping detections to confirm a face
        minSize=(30, 30)    # Rejects objects smaller than 30x30 pixels
    )

    print(f"[SUCCESS] Detected {len(faces)} potential face area configurations.")

    # 5. Draw bounding boxes over the verified regions
    for (x, y, w, h) in faces:
        # Draw a Green box surrounding the discovered face perimeter
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Overlay an identifying text label above the box matrix bounds
        cv2.putText(image, "FACE", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 6. Save structural mapping results to drive storage
    output_path = "detected_faces_output.png"
    cv2.imwrite(output_path, image)
    print(f"Saved structural bounding layout directly to '{output_path}'.")

if __name__ == "__main__":
    main()