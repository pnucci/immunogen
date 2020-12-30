from tensorflow import keras
from tensorflow.keras import layers, utils
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from dataset import X, num_outputs
from sklearn.pipeline import Pipeline
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam


def build_model(learning_rate):
    input_shape = (*X[0].shape, 1)
    model = keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=2, activation="relu",
                      padding='same', input_shape=input_shape),
        layers.Dropout(0.5),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(units=num_outputs, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        # verbose=0
    )
    return model


def reshape_3d(x):
    return x.reshape(*x.shape, 1)


pipeline = Pipeline([
    ('reshape', FunctionTransformer(func=reshape_3d)),
    ('model', KerasClassifier(build_fn=build_model, verbose=0)),
])

params = dict(
    model__batch_size=[25, 50, 100],
    model__epochs=[10, 20, 50],
    model__learning_rate=[0.01, 0.005, 0.001],
)
