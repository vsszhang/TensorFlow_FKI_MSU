"""
Traning convolution netural network with CIFAR-10 model
"""

#import tensorflow as tf
import keras
from keras import layers
from pathlib import Path

def build_model():
    model =keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        layers.Rescaling(1/255.),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    # upload CIFAR-10 data set
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    model = build_model()
    model.summary()

    # Training model (n epochs)
    model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.1)

    print("\n[Test accuracy]")
    print(model.evaluate(x_test, y_test, verbose=2))

    Path("models").mkdir(exist_ok=True)
    model.save("models/cifar10_cnn.keras")