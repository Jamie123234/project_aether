import cv2
import os

# --- Configuration ---
# The folder where your frames are stored
FRAMES_FOLDER = "frames"

# Choose two consecutive frames to compare
FRAME_A_NUM = 150
FRAME_B_NUM = 151
# -------------------

def find_pixel_changes():
    """Loads two frames, finds the differences, and displays them."""

    # Construct the full file paths for the two frames
    frame_a_path = os.path.join(FRAMES_FOLDER, f"frame_{str(FRAME_A_NUM).zfill(5)}.jpg")
    frame_b_path = os.path.join(FRAMES_FOLDER, f"frame_{str(FRAME_B_NUM).zfill(5)}.jpg")

    # Check if the files exist
    if not os.path.exists(frame_a_path) or not os.path.exists(frame_b_path):
        print(f"Error: Make sure both {frame_a_path} and {frame_b_path} exist.")
        return

    # Read the images in grayscale
    frame_a = cv2.imread(frame_a_path, cv2.IMREAD_GRAYSCALE)
    frame_b = cv2.imread(frame_b_path, cv2.IMREAD_GRAYSCALE)

    # --- UPDATED ---
    # Apply a Gaussian blur to each frame to reduce noise
    frame_a_blur = cv2.GaussianBlur(frame_a, (5, 5), 0)
    frame_b_blur = cv2.GaussianBlur(frame_b, (5, 5), 0)

    # Calculate the absolute difference between the two blurred frames
    diff = cv2.absdiff(frame_a_blur, frame_b_blur)
    
    # --- UPDATED ---
    # Apply adaptive Otsu's threshold to automatically find the best contrast
    _, diff_thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Show the resulting difference image in a new window
    print("Displaying difference image. Press any key to close.")
    cv2.imshow(f"Difference between frame {FRAME_A_NUM} and {FRAME_B_NUM}", diff_thresh)
    
    # Wait for the user to press a key, then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    find_pixel_changes()