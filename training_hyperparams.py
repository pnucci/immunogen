# %%
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
from dataset import X, y
import cnn_solution
import lstm_solution

RANDOM_STATE = 0

solution = cnn_solution
print(pd.Series(y).value_counts())

# %%

# finds the best model for this dataset
# runs hyperparam search (the inner CV) in the whole dataset

cv_hyperparams = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=RANDOM_STATE
)

hyperparam_search = RandomizedSearchCV(
    estimator=solution.pipeline,
    n_iter=20,
    scoring=['average_precision', 'roc_auc'],
    param_distributions=solution.params,
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
csv_path = os.path.join(results_path, 'hyperparams.csv')
model_path = os.path.join(results_path, 'best_estimator.pickle')
results_df.to_csv(csv_path)
with open(model_path, 'wb') as out:
    pickle.dump(results.best_estimator_, out)
