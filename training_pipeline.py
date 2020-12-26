# %%
%load_ext autoreload
%autoreload 2
import seaborn as sn
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, utils
import numpy as np
import dataset
import matplotlib.pyplot as plt
import pandas as pd
import cnn_solution
import lstm_solution
# %%

df = dataset.df.sample(frac=1)
print(len(df))

split_frac = 0.7
split_ix = int(split_frac*len(df))
train = df[:split_ix]
test = df[split_ix:]

print(train.y.value_counts())
print(test.y.value_counts())

train_X = np.stack(train.X.values)
train_y = utils.to_categorical(train.y.values)
test_X = np.stack(test.X.values)
test_y = utils.to_categorical(test.y.values)

# LSTM
# input_shape = train.X.iloc[0].shape

# CNN
train_X = train_X.reshape(*train_X.shape, 1)
test_X = test_X.reshape(*test_X.shape, 1)
input_shape = (*train.X.iloc[0].shape, 1)

print('train_X.shape', train_X.shape)
print('input_shape', input_shape)
print(train_X.shape, train_y.shape)
print(test_X.shape, test_y.shape)
classes = np.unique(train_y)
num_outputs = len(classes)
print('num_outputs', num_outputs)

model = cnn_solution.build_model(input_shape)
# model = lstm_solution.build_model(input_shape)
model.add(
    layers.Dense(units=num_outputs, activation='softmax')
)

aupr = keras.metrics.AUC(
    curve="PR",
    name='aupr',
)
auroc = keras.metrics.AUC(
    curve="ROC",
    name='auroc',
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[aupr, auroc],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
BATCH_SIZE = 64

history = model.fit(
    train_X, train_y,
    validation_data=(test_X, test_y),
    batch_size=BATCH_SIZE,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=1,
)

preds = model.predict(test_X, batch_size=BATCH_SIZE)
y_predict_non_category = [np.argmax(t) for t in preds]
test_y_non_category = [np.argmax(t) for t in test_y]
pred_df = pd.DataFrame({
    'pred': y_predict_non_category,
    'label': test_y_non_category,
})


# %%
# plt.figure(figsize=(10, 7))
df_cm = pd.DataFrame(
    confusion_matrix(test_y_non_category, y_predict_non_category),
    index=[classes],
    columns=[classes]
)
sn.heatmap(
    df_cm,
    annot=True,
    cmap=plt.cm.Blues,
    fmt='g'
)
plt.ylabel('True')
plt.xlabel('Predicted')

