from tensorflow import keras
from tensorflow.keras import layers, utils
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from original_approach import dataset
from sklearn.pipeline import Pipeline
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from skopt.space import Real, Categorical, Integer

X = dataset.X
y = dataset.y


def build_model(
    learning_rate,
    blocks,
    kernel_size,
    initial_filters,
    dropout
):
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
    model.add(layers.Dense(units=dataset.num_outputs, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
    )
    print(input_shape)
    model.summary()
    return model


def reshape_3d(x):
    return x.reshape(*x.shape, 1)


estimator = Pipeline([
    ('reshape', FunctionTransformer(func=reshape_3d)),
    ('model', KerasClassifier(build_fn=build_model, verbose=0)),
])

param_distributions_original = dict(
    model__batch_size=[50, 100, 200],
    model__epochs=[20, 50],
    model__learning_rate=[0.001, 0.005],
    model__blocks=[1, 2],
    model__kernel_size=[2, 3],
    model__initial_filters=[30, 60],
    model__dropout=[0.2, 0.4],
)


param_distributions = dict(
    model__batch_size=[50, 100, 200],
    model__epochs=[20, 40, 60, 90],
    model__learning_rate=[0.001, 0.003, 0.005],
    model__blocks=[1, 2, 3, 4],
    model__kernel_size=[2, 3, 4],
    model__initial_filters=[200, 300, 500],
    model__dropout=[0.2, 0.3, 0.4, 0.5, 0.6],
)

search_spaces = dict(
    model__batch_size=Integer(10, 300),
    model__epochs=Integer(5, 50),
    model__learning_rate=Real(0.0001, 0.01, prior='log-uniform'),
    model__blocks=Integer(1, 3),
    model__kernel_size=Integer(2, 5),
    model__initial_filters=Integer(10, 500),
    model__dropout=Real(0.2, 0.4),
)
