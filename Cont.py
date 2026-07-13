import numpy as np
import cv2

def create_binary_shapes():
    """Generates a clean black-and-white binary baseline canvas."""
    # Create a 500x500 flat black background canvas
    img = np.zeros((500, 500), dtype=np.uint8)
    # Add a white square foreground object
    cv2.rectangle(img, (60, 80), (220, 240), (255,), -1)
    # Add a separate white circle foreground object
    cv2.circle(img, (360, 320), 70, (255,), -1)
    return img

def main():
    # 1. Initialize clean binary mask data
    binary_src = create_binary_shapes()
    cv2.imwrite("0_binary_mask.png", binary_src)
    print("Generated binary test canvas layout.")

    # Convert to standard 3-channel color layout so we can draw colorful details
    color_output = cv2.cvtColor(binary_src, cv2.COLOR_GRAY2BGR)

    # 2. Extract structural boundaries
    # Syntax: contours, hierarchy = cv2.findContours(image, mode, method)
    contours, hierarchy = cv2.findContours(
        binary_src, 
        cv2.RETR_EXTERNAL,      # Only grab external parent outermost shapes
        cv2.CHAIN_APPROX_SIMPLE # Compress horizontal/vertical segments to endpoints
    )
    print(f"Successfully identified {len(contours)} distinct shapes.")

    # 3. Process and loop over each discovered boundary
    for i, cnt in enumerate(contours):
        # A. Draw the raw extracted continuous contour line path
        # Color: Green (0, 255, 0), Thickness: 3 pixels
        cv2.drawContours(color_output, [cnt], -1, (0, 255, 0), 3)
        
        # B. Compute individual bounding box perimeter coordinates
        # Returns: (top_left_x, top_left_y, width, height)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # C. Draw a bounding box frame surrounding the components
        # Color: Red (0, 0, 255), Thickness: 2 pixels
        cv2.rectangle(color_output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # D. Stamp index metadata text identifiers over the object positions
        cv2.putText(
            color_output, 
            f"ID: {i+1}", 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )

    # 4. Export the mapped drawing layout
    cv2.imwrite("1_contours_mapped.png", color_output)
    print("Saved mapped boundary details to '1_contours_mapped.png'.")

if __name__ == "__main__":
    main()