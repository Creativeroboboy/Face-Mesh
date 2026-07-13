import sys
import time
import os
import numpy as np
import cv2


def cv2_gui_available() -> bool:
    try:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow("test", np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow("test")
        return True
    except cv2.error:
        return False
    except Exception:
        return False


def main():
    # 1. Initialize camera stream. 
    # Swap '0' for a "video_file.mp4" path string if processing raw files.
    cap = cv2.VideoCapture(0)
    
    # Track framework status to determine if we use the webcam or synthetic fallback
    using_camera = cap.isOpened()
    if not using_camera:
        print("[WARNING] Hardware camera not detected. Running simulation mode with fallback frames.")
    else:
        print("[SUCCESS] Live camera stream connected.")
        print("-> Hold a green object in front of the lens to see it tracked.")
    
    print("-> Press 'q' inside a video window frame to exit.")

    gui_available = cv2_gui_available()
    if not gui_available:
        print("[WARNING] OpenCV GUI support is unavailable in this environment.")
        print("[INFO] The tracker will save one frame to disk instead of opening a window.")

    # 2. Define the Target Color Boundaries (HSV Space)
    # OpenCV Hue range is 0-180. These ranges target green surfaces.
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    # Simulation state counter
    sim_frame_idx = 0

    # 3. Enter the Processing Loop
    while True:
        if using_camera:
            ret, frame = cap.read()
            if not ret:
                print("Error: Video stream interrupted.")
                break
        else:
            # Fallback Simulation: Generate a canvas containing a moving green square
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 40
            sim_frame_idx = (sim_frame_idx + 4) % 400
            # Draw moving target green square (BGR structure: Blue=0, Green=200, Red=0)
            cv2.rectangle(frame, (100 + sim_frame_idx, 150), (220 + sim_frame_idx, 270), (0, 200, 0), -1)
            # Draw an irrelevant blue distracting square
            cv2.rectangle(frame, (50, 300), (120, 370), (220, 0, 0), -1)

        # 4. Color Transformation
        # Convert frame color space layout from default BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 5. Segment Color Target via Masking
        # Creates a binary image: white pixels are within the range, black pixels are out
        mask = cv2.inRange(hsv, lower_red1, upper_red1)

        # 6. Smooth the Mask Array (Erase floating noise specs)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # 7. Contour Detection and Tracking Analysis
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any green targets exist
        if len(contours) > 0:
            # Sort through outlines to find the largest spatial bounding region
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Eliminate microscopic background glare noise
            if cv2.contourArea(largest_contour) > 400:
                # Compute bounding frame values
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Draw tracker bounds directly over original frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
                
                # Stamp status notification labels
                cv2.putText(frame, "TRACKING GREEN TARGET", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 8. Display Output States
        if gui_available:
            cv2.imshow("Object Tracker - Raw Source Feed", frame)
            cv2.imshow("Object Tracker - Isolate HSV Mask", mask)

            # 9. Key Exit Command Listeners
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("Exit signal received. Cleaning down locks...")
                break
        else:
            output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ct_track_output")
            os.makedirs(output_folder, exist_ok=True)
            timestamp = int(time.time() * 1000)
            raw_path = os.path.join(output_folder, f"raw_{timestamp}.png")
            mask_path = os.path.join(output_folder, f"mask_{timestamp}.png")
            cv2.imwrite(raw_path, frame)
            cv2.imwrite(mask_path, mask)
            print(f"GUI unavailable: saved frame to {raw_path} and {mask_path}")
            print("Set GUI support or open the saved images manually. Exiting after one frame.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()