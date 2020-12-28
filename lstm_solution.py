from tensorflow import keras
from dataset import dataset

preprocess = []

def build_model():
    input_shape = dataset.X.iloc[0].shape
    return keras.Sequential([
        keras.layers.LSTM(50, input_shape=input_shape, return_sequences=True),
        keras.layers.LSTM(64),
        keras.layers.Dense(64),
        keras.layers.Dropout(0.5)
])
