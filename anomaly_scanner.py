import os
import cv2
import numpy as np
import tensorflow as tf
import csv

# --- Configuration ---
MODEL_PATH = "anomaly_detector.keras"
FRAMES_DIR = "frames"
OUTPUT_CSV = "anomalies.csv"
IMAGE_SIZE = (64, 64)
# --- NEW: Tunable Parameter ---
# The minimum length (in pixels) of a streak to be considered a potential meteor.
# Increase this value if you are still detecting too many star trails.
MIN_STREAK_LENGTH = 30
# -----------------------------

def scan_frames_for_anomalies():
    print("Loading trained anomaly detector model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
    
    if len(frame_files) < 2:
        print("Error: Not enough frames in the directory to compare.")
        return

    found_anomalies = []
    print(f"Starting scan of {len(frame_files)} frames...")

    for i in range(len(frame_files) - 1):
        frame_a_path = os.path.join(FRAMES_DIR, frame_files[i])
        frame_b_path = os.path.join(FRAMES_DIR, frame_files[i+1])

        frame_a_gray = cv2.imread(frame_a_path, cv2.IMREAD_GRAYSCALE)
        frame_b_gray = cv2.imread(frame_b_path, cv2.IMREAD_GRAYSCALE)
        frame_b_color = cv2.imread(frame_b_path)

        if frame_a_gray is None or frame_b_gray is None:
            continue
        
        frame_a_blur = cv2.GaussianBlur(frame_a_gray, (5, 5), 0)
        frame_b_blur = cv2.GaussianBlur(frame_b_gray, (5, 5), 0)

        diff = cv2.absdiff(frame_a_blur, frame_b_blur)
        _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # --- UPDATED LOGIC ---
            # This is the new filter. We check if the bounding box of the
            # change is long enough in either width or height to be a streak.
            if w < MIN_STREAK_LENGTH and h < MIN_STREAK_LENGTH:
                continue # Ignore this contour as it's too small (likely a star trail).
            # ---------------------

            roi = frame_b_color[y:y+h, x:x+w]
            
            img_resized = cv2.resize(roi, IMAGE_SIZE)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_array = tf.keras.utils.img_to_array(img_gray)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)
            score = prediction[0][0]

            if score < 0.5:
                frame_number = i + 1
                center_x = x + w // 2
                center_y = y + h // 2
                found_anomalies.append([frame_number, center_x, center_y])
                print(f"  -> Potential Anomaly found in frame {frame_number} at ({center_x}, {center_y}) with score {score:.2f}")

    print(f"\nScan complete. Found {len(found_anomalies)} potential anomalies.")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_number', 'x', 'y'])
        writer.writerows(found_anomalies)

    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    scan_frames_for_anomalies()