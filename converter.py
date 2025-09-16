# Import the necessary libraries
import cv2
import os

# --- Configuration ---
# Define the path to your input video
VIDEO_PATH = "videos/NGT026.mp4"

# Define the folder where you want to save the frames
OUTPUT_FOLDER = "frames"
# -------------------

def video_to_frames():
    """Reads a video file and saves each frame as a separate JPG image."""

    # Create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created directory: {OUTPUT_FOLDER}")

    # Open the video file
    video_capture = cv2.VideoCapture(VIDEO_PATH)

    # Check if the video opened successfully
    if not video_capture.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        return

    print("Starting video to frame conversion...")
    frame_count = 0

    # Loop through the video frames
    while True:
        # Read one frame from the video
        success, frame = video_capture.read()

        # If 'success' is False, we have reached the end of the video
        if not success:
            break

        # Construct the output filename with leading zeros (e.g., frame_00001.jpg)
        file_name = f"frame_{str(frame_count).zfill(5)}.jpg"
        file_path = os.path.join(OUTPUT_FOLDER, file_name)

        # Save the current frame as a JPG file
        cv2.imwrite(file_path, frame)

        # Print progress to the console
        if frame_count % 30 == 0: # Print every 30 frames
             print(f"Saved {file_name}")

        frame_count += 1

    # Release the video capture object to free up resources
    video_capture.release()
    print("-" * 20)
    print(f"Conversion complete. Total frames saved: {frame_count}")
    print(f"Frames are located in the '{OUTPUT_FOLDER}' directory.")
    print("-" * 20)


# Run the main function when the script is executed
if __name__ == "__main__":
    video_to_frames()