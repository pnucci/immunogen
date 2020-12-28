# %%
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import seaborn as sn
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, utils
import numpy as np
from dataset import dataset
import matplotlib.pyplot as plt
import pandas as pd
import cnn_solution
import lstm_solution
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import average_precision_score, make_scorer
%load_ext autoreload
%autoreload 2

solution = lstm_solution

RANDOM_SEED = 0
N_SPLITS = 5
N_REPEATS = 2
early_stop_metric = 'val_loss'

# %%

print(dataset['y'].value_counts())

test_split = StratifiedShuffleSplit(
    n_splits=1, test_size=0.3).split(dataset.X, dataset.y)
train_ix, test_ix = list(test_split)[0]
train = dataset.iloc[train_ix]
test = dataset.iloc[test_ix]

# %%

train_prop = train.y.value_counts()/len(train)
test_prop = test.y.value_counts()/len(test)
class_proportion_preserved = np.allclose(train_prop, test_prop, atol=0.01)
assert(class_proportion_preserved)

# %%

train_X = np.stack(train.X.values)
train_y = train.y.values
train_y_cat = utils.to_categorical(train.y.values)
test_X = np.stack(test.X.values)
test_y = test.y.values
test_y_cat = utils.to_categorical(test.y.values)

print('train_X.shape', train_X.shape)
print(train_X.shape, train_y.shape)
print(test_X.shape, test_y.shape)
classes = np.unique(train_y)
num_outputs = len(classes)
print('num_outputs', num_outputs)

aupr = keras.metrics.AUC(
    curve="PR",
    name='aupr',
)
auroc = keras.metrics.AUC(
    curve="ROC",
    name='auroc',
)


def build_model():
    model = solution.build_model()
    model.add(
        layers.Dense(units=num_outputs, activation='softmax')
    )
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[aupr, auroc],
    )
    return model


# %%
pipeline = Pipeline([
    *solution.preprocess,
    ('model', KerasClassifier(
        build_fn=build_model,
        verbose=1
    ))
])
kfold = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_SEED
)
hyperparam_search = RandomizedSearchCV(
    estimator=pipeline,
    n_iter=2,
    scoring=['average_precision', 'roc_auc'],
    param_distributions=dict(
        model__batch_size=[50, 100],
        model__epochs=[20, 30]
    ),
    refit='roc_auc',
    # n_jobs=-1,
    cv=kfold
)
hyperparam_search_result = hyperparam_search.fit(train_X, train_y)

# %%
print("Best: %f using %s" % (hyperparam_search_result.best_score_,
                             hyperparam_search_result.best_params_))
means = hyperparam_search_result.cv_results_['mean_test_roc_auc']
stds = hyperparam_search_result.cv_results_['std_test_roc_auc']
params = hyperparam_search_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
results_df = pd.DataFrame(hyperparam_search_result.cv_results_)
results_df.sort_values('rank_test_roc_auc')[
    ['params', 'mean_test_roc_auc', 'std_test_roc_auc']]
# %%
best = hyperparam_search_result.best_estimator_
preds = best.predict_proba(test_X)
preds
# %%
roc_auc_score(test_y_cat, preds)
# %%
