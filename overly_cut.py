import cv2
import numpy as np

# Load images
motion_vector = cv2.imread("output/motion_vectors/frame_0001.png", cv2.IMREAD_GRAYSCALE)
refined_mask = cv2.imread("output/refined_masks/frame_0001.png", cv2.IMREAD_GRAYSCALE)

# Ensure both images are the same size
if motion_vector.shape != refined_mask.shape:
    refined_mask = cv2.resize(refined_mask, (motion_vector.shape[1], motion_vector.shape[0]), interpolation=cv2.INTER_NEAREST)

# Overlay the refined mask onto the motion vector frame
overlay = cv2.addWeighted(motion_vector, 0.5, refined_mask, 0.5, 0)

# Save and show
cv2.imwrite("output/debug_overlay.png", overlay)
cv2.imshow("Refined Mask Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
