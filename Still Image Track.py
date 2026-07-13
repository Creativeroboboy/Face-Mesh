import os
import cv2
from ultralytics import YOLO


def process_static_image():
    # Load a pre-trained model (can swap with "yolov8n.pt")
    model = YOLO("yolo11n.pt")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(base_dir, "face.jpg")
    output_path = os.path.join(base_dir, "image_predictions.png")

    # Run inference explicitly using predict mode
    results = model.predict(source=image_path, conf=0.25)[0]

    # Load original image with OpenCV to overlay bounding structures manually
    img = cv2.imread(image_path)

    print("\n--- Detected Object Attributes ---")
    # Loop over every detected structural bounding box
    for box in results.boxes:
        # 1. Bounding Box Coordinates (xyxy format: top-left x, y; bottom-right x, y)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # 2. Confidence Score (float representation between 0.0 and 1.0)
        confidence = float(box.conf[0])

        # 3. Object Class ID and Descriptive Label String
        class_id = int(box.cls[0])
        label = model.names[class_id]

        print(
            f"Label: {label} | Conf: {confidence:.2f} | Box: [{x1}, {y1}, {x2}, {y2}]"
        )

        # Draw structural rectangle using OpenCV array mappings
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Build annotations string
        caption = f"{label} {confidence:.2f}"
        cv2.putText(
            img,
            caption,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Save output frame tracking result to disk
    cv2.imwrite(output_path, img)
    print(f"\nProcessing complete! Output saved to '{output_path}'")


if __name__ == "__main__":
    process_static_image()