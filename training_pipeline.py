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

# outer cv
cv_oos_performance = RepeatedStratifiedKFold(
    n_splits=3,
    n_repeats=1,
    random_state=RANDOM_SEED
)

# inner cv
cv_hyperparams = RepeatedStratifiedKFold(
    n_splits=3,
    n_repeats=1,
    random_state=RANDOM_SEED
)


print(dataset['y'].value_counts())

X = np.stack(dataset.X.values)
y = dataset.y.values
y_cat = utils.to_categorical(y)

num_outputs = len(np.unique(y))
print('num_outputs', num_outputs)


def build_model(hidden_layers, learning_rate) -> KerasClassifier:
    print(learning_rate)
    model = solution.build_model()
    model.add(
        layers.Dense(units=num_outputs, activation='softmax')
    )
    from tensorflow.keras.optimizers import Adam
    model.compile(
        # optimizer='adam',
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.AUC(curve="PR", name='aupr'),
            keras.metrics.AUC(curve="ROC", name='auroc')
        ],
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

hyperparam_search = RandomizedSearchCV(
    estimator=pipeline,
    n_iter=2,
    scoring=['average_precision', 'roc_auc'],
    param_distributions=dict(
        model__batch_size=[25, 50, 100],
        model__epochs=[10, 20, 50],
        model__hidden_layers=[20],
        model__learning_rate=[0.01,0.005,0.001]

    ),
    refit='roc_auc',
    # n_jobs=-1,
    cv=cv_hyperparams
)
# %%
oos_performance = cross_validate(
    estimator=hyperparam_search,
    X=X,
    y=y,
    cv=cv_oos_performance,
    return_estimator=True
)
#%%
print(oos_performance)
pd.DataFrame(oos_performance)

# hyperparam_search_result = hyperparam_search.fit(train_X, train_y)
# # %%
# print("Best: %f using %s" % (hyperparam_search_result.best_score_,
#                              hyperparam_search_result.best_params_))
# means = hyperparam_search_result.cv_results_['mean_test_roc_auc']
# stds = hyperparam_search_result.cv_results_['std_test_roc_auc']
# params = hyperparam_search_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
# results_df = pd.DataFrame(hyperparam_search_result.cv_results_)
# results_df.sort_values('rank_test_roc_auc')[
#     ['params', 'mean_test_roc_auc', 'std_test_roc_auc']]
# # %%
# best = hyperparam_search_result.best_estimator_
# preds = best.predict_proba(test_X)
# preds
# # %%
# roc_auc_score(test_y_cat, preds)
