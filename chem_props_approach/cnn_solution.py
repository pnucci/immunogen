from tensorflow import keras
from tensorflow.keras import layers, utils
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from chem_props_approach.dataset import X, num_outputs
from sklearn.pipeline import Pipeline
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam

previous = {}

def reshape_before(X):
    previous['shape'] = X.shape
    return X.reshape(-1, X.shape[-1])


def reshape_after(X):
    X = X.reshape(previous['shape'])
    return X

def build_model(
    learning_rate,
    blocks,
    kernel_size,
    initial_filters,
    dropout
):
    print(locals())
    input_shape = X[0].shape
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_shape)))
    for block in range(blocks):
        filters = initial_filters*(block+1)
        model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="relu",
                                padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool1D())
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

pipeline = Pipeline([
    ('reshape_before', FunctionTransformer(func=reshape_before)),
    ('scale', StandardScaler()),
    ('reshape_after', FunctionTransformer(func=reshape_after)),
    ('model', KerasClassifier(build_fn=build_model, verbose=0)),
])

params = dict(
    model__batch_size=[50, 100, 200],
    model__epochs=[20, 40, 60, 90],
    model__learning_rate=[0.001, 0.003, 0.005],
    model__blocks=[1, 2, 3, 4],
    model__kernel_size=[2, 3, 4],
    model__initial_filters=[200, 300, 500],
    model__dropout=[0.2, 0.3, 0.4, 0.5, 0.6],
)
