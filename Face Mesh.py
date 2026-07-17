import cv2
import numpy as np

def run_face_mesh():
    # Load pre-trained face detector and facial landmarks detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=4,
            minSize=(30, 30)
        )
        
        # Draw face rectangles and simple mesh
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw a simple mesh/grid inside the face
            grid_spacing_x = w // 4
            grid_spacing_y = h // 4
            
            # Draw vertical lines
            for i in range(1, 4):
                pt1 = (x + i * grid_spacing_x, y)
                pt2 = (x + i * grid_spacing_x, y + h)
                cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
            
            # Draw horizontal lines
            for i in range(1, 4):
                pt1 = (x, y + i * grid_spacing_y)
                pt2 = (x + w, y + i * grid_spacing_y)
                cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
            
            # Draw key points (corner and center points as placeholders for landmarks)
            key_points = [
                (x, y),                    # top-left
                (x + w, y),                # top-right
                (x, y + h),                # bottom-left
                (x + w, y + h),            # bottom-right
                (x + w // 2, y + h // 2),  # center
                (x + w // 2, y),           # top-center
                (x + w // 2, y + h),       # bottom-center
                (x, y + h // 2),           # left-center
                (x + w, y + h // 2),       # right-center
            ]
            
            for point in key_points:
                cv2.circle(frame, point, 3, (0, 255, 255), -1)
        
        cv2.imshow("Face Mesh Detection", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_mesh()