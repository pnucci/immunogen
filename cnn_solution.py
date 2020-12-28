from tensorflow import keras
from tensorflow.keras import layers, utils
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from dataset import dataset

def reshape_3d(x):
    return x.reshape(*x.shape, 1)


preprocess = [
    ('reshape3d', FunctionTransformer(func=reshape_3d))
]


def build_model():
    input_shape = (*dataset.X.iloc[0].shape, 1)    
    return keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=2, activation="relu",
                      padding='same', input_shape=input_shape),
        layers.Dropout(0.5),
        layers.MaxPool2D(),
        layers.Flatten(),
    ])
