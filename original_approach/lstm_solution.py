from sklearn.pipeline import Pipeline
from tensorflow.keras import layers, utils, Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from original_approach import dataset

X = dataset.X
y = dataset.y

def build_model(learning_rate):
    input_shape = X[0].shape
    model = Sequential([
        layers.LSTM(50, input_shape=input_shape, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(64),
        layers.Dropout(0.5),
        layers.Dense(units=dataset.num_outputs, activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy'
    )
    model.summary()
    return model


estimator = Pipeline([
    ('model', KerasClassifier(build_fn=build_model, verbose=0)),
])

param_distributions = dict(
    model__batch_size=[25, 50, 100],
    model__epochs=[10, 20, 50],
    model__learning_rate=[0.01, 0.005, 0.001],
)
