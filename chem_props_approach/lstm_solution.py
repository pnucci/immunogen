from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, utils, Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from chem_props_approach import dataset
from sklearn.preprocessing import FunctionTransformer

X = dataset.X
y = dataset.y

_previous_state = {}

def reshape_before(X):
    _previous_state['shape'] = X.shape
    return X.reshape(-1, X.shape[-1])


def reshape_after(X):
    X = X.reshape(_previous_state['shape'])
    return X


def build_model(learning_rate, dropout, nodes_per_layer):
    input_shape = X[0].shape
    model = Sequential([
        layers.LSTM(nodes_per_layer, input_shape=input_shape,
                    return_sequences=True),
        layers.LSTM(nodes_per_layer),
        layers.Dense(nodes_per_layer),
        layers.Dropout(dropout),
        layers.Dense(units=dataset.num_outputs, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy'
    )
    model.summary()
    return model


estimator = Pipeline([
    ('reshape_before', FunctionTransformer(func=reshape_before)),
    ('scale', StandardScaler()),
    ('reshape_after', FunctionTransformer(func=reshape_after)),
    ('model', KerasClassifier(build_fn=build_model, verbose=0)),
])

param_distributions = dict(
    model__batch_size=[25, 35, 50, 100],
    model__epochs=[10, 20, 30, 50],
    model__learning_rate=[0.005, 0.003, 0.001, 0.0005],
    model__dropout=[0.2, 0.3, 0.4],
    model__nodes_per_layer=[200, 400, 500, 600, 800]
)
