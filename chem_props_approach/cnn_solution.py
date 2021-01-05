from chem_props_approach import dataset
from tensorflow import keras
from tensorflow.keras import layers, utils
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam

X = dataset.X
y = dataset.y

_input_shape = X[0].shape
_previous_state = {}

def _reshape_before(X):
    _previous_state['shape'] = X.shape
    return X.reshape(-1, X.shape[-1])


def _reshape_after(X):
    X = X.reshape(_previous_state['shape'])
    return X

def _build_model(
    learning_rate,
    blocks,
    kernel_size,
    initial_filters,
    dropout
):
    model = keras.Sequential()
    model.add(keras.Input(shape=(_input_shape)))
    for block in range(blocks):
        filters = initial_filters*(block+1)
        model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="relu",
                                padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool1D())
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(units=dataset.num_outputs, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
    )
    model.summary()
    return model

estimator = Pipeline([
    ('reshape_before', FunctionTransformer(func=_reshape_before)),
    ('scale', StandardScaler()),
    ('reshape_after', FunctionTransformer(func=_reshape_after)),
    ('model', KerasClassifier(build_fn=_build_model, verbose=0)),
])

param_distributions = dict(
    model__batch_size=[50, 100, 200],
    model__epochs=[20, 40, 60, 90],
    model__learning_rate=[0.001, 0.003, 0.005],
    model__blocks=[1, 2, 3, 4],
    model__kernel_size=[2, 3, 4],
    model__initial_filters=[200, 300, 500],
    model__dropout=[0.2, 0.3, 0.4, 0.5, 0.6],
)
