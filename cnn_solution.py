from tensorflow import keras
from tensorflow.keras import layers, utils
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from dataset import X, num_outputs
from sklearn.pipeline import Pipeline
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam


def build_model(
    learning_rate,
    blocks,
    kernel_size,
    initial_filters,
    dropout
):
    print(locals())
    input_shape = (*X[0].shape, 1)
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_shape)))
    for block in range(blocks):
        filters = initial_filters*(block+1)
        model.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation="relu",
                                padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D())
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(units=num_outputs, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
    )
    print(input_shape)
    model.summary()
    return model


def reshape_3d(x):
    return x.reshape(*x.shape, 1)


pipeline = Pipeline([
    ('reshape', FunctionTransformer(func=reshape_3d)),
    ('model', KerasClassifier(build_fn=build_model, verbose=0)),
])

params = dict(
    model__batch_size=[50, 100, 200],
    model__epochs=[20, 50],
    model__learning_rate=[0.001, 0.005],
    model__blocks=[1, 2],
    model__kernel_size=[2, 3],
    model__initial_filters=[30, 60],
    model__dropout=[0.2, 0.4],
)
