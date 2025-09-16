import tensorflow as tf
from tensorflow import keras
from keras import layers
import os

# --- Configuration ---
# This is the directory where your 'anomaly' and 'normal_sky' folders are.
DATASET_DIR = "dataset" 

# All your training images will be resized to this dimension.
IMAGE_SIZE = (64, 64) 

# How many images the AI looks at in one go.
BATCH_SIZE = 16 

# How many times the AI will study the entire dataset.
EPOCHS = 10 
# -------------------

def build_and_train_model():
    """Loads the dataset, builds a CNN model, trains it, and saves it."""

    print("Loading training and validation datasets...")
    # Automatically load images from the folders and split them
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale'
    )

    class_names = train_ds.class_names
    print(f"Found classes: {class_names}")

    # --- Build the AI Model (CNN) ---
    print("Building the CNN model...")
    model = keras.Sequential([
        layers.Rescaling(1./255),
        
        # --- NEW: DATA AUGMENTATION ---
        # Create slightly modified versions of images during training
        # to make the model more robust.
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        # --------------------------------

        # First convolutional layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        # Second convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        # Flatten the data into a single line.
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        # Final output layer
        layers.Dense(1, activation='sigmoid')
    ])
    # --------------------------------

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Save the fully trained model to a single file.
    model.save("anomaly_detector.keras")
    print("-" * 20)
    print("Training complete. Model saved as 'anomaly_detector.keras'")
    print("-" * 20)


if __name__ == "__main__":
    try:
        if not os.path.exists(DATASET_DIR) or not os.listdir(os.path.join(DATASET_DIR, "anomaly")) or not os.listdir(os.path.join(DATASET_DIR, "normal_sky")):
            print(f"Error: The '{DATASET_DIR}' directory is not set up correctly.")
            print("Please create 'anomaly' and 'normal_sky' subdirectories and add your cropped images to them.")
        else:
            build_and_train_model()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your dataset folder is structured correctly with 'anomaly' and 'normal_sky' subfolders.")