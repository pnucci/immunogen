from tensorflow import keras
from tensorflow.keras import layers, utils

def build_model(input_shape):
    return keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=4, activation="relu",
                  padding='same', input_shape=input_shape),
    layers.Dropout(0.4),
    layers.MaxPool2D(),
    layers.Flatten(),
])
