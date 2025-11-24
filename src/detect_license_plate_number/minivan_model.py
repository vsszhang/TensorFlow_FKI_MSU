"""
    Desc: Train the model
"""

import keras
from keras import layers
from pathlib import Path
import numpy as np
import cv2

BASE = "src/detect_license_plate_number/"

""" Build CNN Model """
def build_model(input_shape=(128, 128, 3), num_classes=3):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

""" Building, Training... (Dataset -> CNN) """
def train_model():
    DATA_DIR = BASE + "Dataset"
    OUT_PATH = Path(BASE + "model/minivan_cnn.keras")
    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        batch_size=32,
        image_size=(128, 128),
        validation_split=0.2,
        subset="training",
        seed=42,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        batch_size=32,
        image_size=(128, 128),
        validation_split=0.2,
        subset="validation",
        seed=42,
    )

    # Print class numbers
    class_names = train_ds.class_names
    print("Class names: ", class_names)

    # Speed data?

    # Build the model
    model = build_model(num_classes=len(class_names))
    model.summary()

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train the model
    model.fit(train_ds, epochs=50, validation_data=val_ds)

    # Save the model
    model.save(OUT_PATH)
    print(f"[INFO] Save model successfuly, saving into: {OUT_PATH}")

""" Load model """
def load_minivan_model(path=f"{BASE}models/minivan_cnn.keras"):
    return keras.models.load_model(path)

""" Predict patch (is 'minivan', 'random' or 'road')"""
def predict_patch(model, patch_bgr):
    """
    Desc: 
    This function will using our trained model to predict every path, from the output
    we can know the patch's type.

    Input:
    model and patch (whatever size, because the OpenCV will resize the path into
    size 128 * 128)
    Output: prediced type (which patch belong to) and the prediced value
    """
    class_list = ["minivan", "random", "road"]

    # resize the patch to size 128 * 128
    patch = cv2.resize(patch_bgr, (128, 128))

    # transfer into `float` (CNN model required)
    patch = patch.astype("float32") / 255.0

    # add batch dimension (1, 128, 128, 3) (CNN model required)
    patch = np.expand_dims(patch, axis=0)

    # model start to predict
    probs = model.predict(patch, verbose=0)[0]
    cls_idx = np.argmax(probs)

    # output the predicted type and predicted value
    return class_list[cls_idx], float(probs[cls_idx])

""" Define command """
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.train:
        train_model()