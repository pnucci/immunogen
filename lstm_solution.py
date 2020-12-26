from tensorflow import keras

def build_model(input_shape):
    return keras.Sequential([
        keras.layers.LSTM(50, input_shape=input_shape, return_sequences=True),
        keras.layers.LSTM(64),
        keras.layers.Dense(64),
        keras.layers.Dropout(0.5)
])
