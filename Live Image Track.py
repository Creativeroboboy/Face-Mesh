import cv2
from ultralytics import YOLO


def process_video_stream():
    # Load pre-trained model
    model = YOLO("yolo11n.pt")

    # For Recorded Video File: Pass the absolute path string, e.g., "traffic.mp4"
    # For Live Webcam Stream: Pass integer 0 or 1 to map system hardware profiles
    video_source = 0

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source index {video_source}")
        return

    print("Streaming initialized. Press 'q' inside video window bounds to exit.")

    # Enter execution loop frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream finished or hardware disconnected.")
            break

        # Run model inference on the raw BGR frame matrix
        # stream=True utilizes generator behavior for optimal performance
        results = model.predict(source=frame, conf=0.30, verbose=False)[0]

        # Extract parsed visual data
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            label = model.names[int(box.cls[0])]

            # Overlay bounding elements dynamically onto camera capture frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        # Render outputs in real-time UI frame windows
        cv2.imshow("Ultralytics YOLO Live Tracking", frame)

        # Break conditional listener
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video_stream()