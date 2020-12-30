# %%
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from dataset import X, y
import cnn_solution
import lstm_solution
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV

RANDOM_SEED = 0

solution = lstm_solution

# outer cv
cv_oos_performance = RepeatedStratifiedKFold(
    n_splits=2,
    n_repeats=1,
    random_state=RANDOM_SEED
)

# inner cv
cv_hyperparams = RepeatedStratifiedKFold(
    n_splits=2,
    n_repeats=1,
    random_state=RANDOM_SEED
)

print(pd.Series(y).value_counts())

hyperparam_search = RandomizedSearchCV(
    estimator=solution.pipeline,
    n_iter=2,
    scoring=['average_precision', 'roc_auc'],
    param_distributions=solution.params,
    refit='roc_auc',
    verbose=10,
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
print('expected oos performance',oos_performance)
pd.DataFrame(oos_performance)

#%%

# discover the best model
# by re-running the inner CV hyperparam search in the whole dataset

results = hyperparam_search.fit(X, y)

#%%
results_df = pd.DataFrame(results.cv_results_)
results_df.sort_values('rank_test_roc_auc')[
    ['params', 'mean_test_roc_auc', 'std_test_roc_auc']]
# %%
