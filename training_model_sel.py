# %%
import re
import winsound
import pickle
import os
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from chem_props_approach.config import X, y, estimator, param_distributions
np.random.seed(0)
RANDOM_STATE = 0

print(pd.Series(y).value_counts())
print(X.shape)

# %%

# finds the best model for this dataset
# runs hyperparam search (the inner CV) in the whole dataset

cv_hyperparams = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=2,
    random_state=RANDOM_STATE
)

hyperparam_search = RandomizedSearchCV(
    estimator=estimator,
    n_iter=50,
    scoring=['average_precision', 'roc_auc'],
    param_distributions=param_distributions,
    refit='roc_auc',  # using the same metric as the paper
    verbose=10,
    # n_jobs=-1,
    cv=cv_hyperparams
)

results = hyperparam_search.fit(X, y)
results_df = pd.DataFrame(results.cv_results_)
results_df = results_df.sort_values('rank_test_roc_auc')
print(results_df[['params', 'mean_test_roc_auc', 'std_test_roc_auc']])

# save results

timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
results_path = os.path.join('tmp', f'training_{timestamp}')
os.makedirs(results_path, exist_ok=True)

# detailed csv

detailed_csv_path = os.path.join(results_path, 'model_selection_detailed.csv')
results_df.to_csv(detailed_csv_path)

# short csv

rxs_to_remove = [
    r'split\d.+',
    r'.+_((fit)|(score))_time',
    r'params'
]


def should_keep(col):
    return not any(re.match(rx, col) for rx in rxs_to_remove)


selected_cols = [c for c in results_df.columns if should_keep(c)]
csv_path = os.path.join(results_path, 'model_selection.csv')
results_df[selected_cols].to_csv(csv_path)

# best model

# model_path = os.path.join(results_path, 'best_estimator.pickle')
# with open(model_path, 'wb') as out:
#     pickle.dump(results.best_estimator_, out)


# %%
winsound.PlaySound("SystemQuestion", winsound.SND_LOOP+winsound.SND_ASYNC)
