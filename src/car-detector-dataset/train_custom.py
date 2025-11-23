'''
    Training model with costom datasat
'''

import keras
from keras import layers
from pathlib import Path

DATA_DIR = "Dataset"
IMG_SIZE = (32, 32)
BATCH_SIZE = 32
EPOCHS = 50

OUT_MODEL = Path("src/car-detector-dataset/models/custom_dataset_cnn.keras")

def load_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Read and output the class names of `Dataset` directory
    class_names = train_ds.class_names
    print("Class name:", class_names)

    return train_ds, val_ds, class_names

def build_model(num_classes: int):
    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),  # (32, 32, 3)
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu"),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0),

        layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def main():
    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)

    model = build_model(num_classes)
    model.summary()

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds
    )

    print("\n[INFO] Training finished.")

    print("\n[INFO] Validation evaluation:")
    model.evaluate(val_ds)

    # Save the model
    model.save(OUT_MODEL)
    print("\n[SAVE] save model at: {OUT_MODEL}")

    # Save the `classes` into a txt file
    classes_txt = OUT_MODEL.with_suffix(".classes.txt")
    with classes_txt.open("w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
        print(f"[SAVE] class names -> {classes_txt}")

if __name__ == "__main__":
    main()