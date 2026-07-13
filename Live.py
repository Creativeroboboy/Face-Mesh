import sys
import cv2


def main():
    # 1. Initialize the video capture source
    # Pass '0' as an integer to select your default system webcam device.
    # Alternatively, pass a string path to target a video file: e.g., "video.mp4"
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    # 2. Verify that the hardware stream or file path opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source at index {camera_index}.")
        sys.exit(1)

    print("Video stream initiated successfully.")
    print("-> Press 'q' while focusing on the video window to exit.")

    # 3. Enter the frame processing loop
    while True:
        # Capture frame-by-frame
        # ret   -> A boolean indicating if the frame was successfully read
        # frame -> The actual frame matrix (a NumPy array in BGR color space)
        ret, frame = cap.read()

        # If ret is False, the video stream has ended or the camera disconnected
        if not ret:
            print("Error: Failed to grab frame or video stream has completed.")
            break

        # ==========================================================
        # REAL-TIME PROCESSING AREA
        # ==========================================================
        # Example 1: Convert the frame live to Grayscale
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Example 2: Draw a red warning circle directly onto the processed frame
        # Syntax: cv2.circle(img, center, radius, color_in_grayscale, thickness)
        cv2.circle(processed_frame, (50, 50), 20, (255,), -1)
        # ==========================================================

        # 4. Display the results in real-time GUI windows
        cv2.imshow("Live Feed - Original", frame)
        cv2.imshow("Live Feed - Processed (Grayscale)", processed_frame)

        # 5. Handle Keyboard Events & Exit Mechanics
        # cv2.waitKey(1) halts execution for 1ms looking for an active key press.
        # Masking with & 0xFF ensures cross-platform compatibility.
        key = cv2.waitKey(1) & 0xFF

        # Check if the user pressed the 'q' key on their keyboard
        if key == ord("q"):
            print("Exit signal received. Shutting down system safely...")
            break

    # 6. Cleanup Resources
    # Always release the hardware camera lock and destroy UI windows to prevent leaks
    cap.release()
    cv2.destroyAllWindows()
    print("Resources cleared. Script finished.")


if __name__ == "__main__":
    main()