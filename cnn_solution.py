from tensorflow import keras
from tensorflow.keras import layers, utils

def build_model(input_shape):
    return keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=2, activation="relu",
                  padding='same', input_shape=input_shape),
    layers.Dropout(0.5),
    layers.MaxPool2D(),
    layers.Flatten(),
])
