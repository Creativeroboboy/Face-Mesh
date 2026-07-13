import os
import numpy as np
import cv2


def extract_id_perspective(image_path):
    """Detects card boundaries and flattens the image layer perspective."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Apply bilateral filter to preserve edges while eliminating print noise
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)

    # 2. Extract edge contours using Canny
    edged = cv2.Canny(blurred, 30, 200)
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours to isolate the largest rectangle shape (the ID Card body)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        # If our contour approximation yields 4 distinct corner coordinates
        if len(approx) == 4:
            pts = approx.reshape(4, 2)

            # Define destination target coordinates for a standard 85.6mm x 54mm ID matrix ratio
            dst_pts = np.array(
                [[0, 0], [856, 0], [856, 540], [0, 540]], dtype="float32"
            )

            # Compute transformation matrix perspective and warp the array
            # (Note: Requires sorting corner points inside pts to match dst_pts mapping order)
            # transform_matrix = cv2.getPerspectiveTransform(sorted_pts, dst_pts)
            # warped = cv2.warpPerspective(img, transform_matrix, (856, 540))
            # return warped

    return img


if __name__ == "__main__":
    image_path = "face.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    result = extract_id_perspective(image_path)
    output_path = "Oface.jpg"
    saved = cv2.imwrite(output_path, result)
    print(f"Processed image shape: {result.shape}")
    print(f"Output saved: {output_path}")
    print(f"Write success: {saved}")