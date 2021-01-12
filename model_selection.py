# Finds the best model for a dataset
# runs hyperparam search (one CV level only) in the whole dataset and saves results
# the best model's score is a BIASED estimate of the model selection procedure's OOS performance

# %%
import importlib
import re
import pickle
import os
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
from copy import copy

with open('model_selection.yaml', 'r') as yamlfile:
    config = yaml.safe_load(yamlfile)
print(config)

solution = importlib.import_module(config['solution_module'])

np.random.seed(config['random_seed'])
RANDOM_STATE = config['random_state']

print(pd.Series(solution.y).value_counts())
print(solution.X.shape)

# %%

cv_hyperparams = RepeatedStratifiedKFold(
    n_splits=config['hyperparam_search']['n_splits'],
    n_repeats=config['hyperparam_search']['n_repeats'],
    random_state=RANDOM_STATE
)

hyperparam_search = BayesSearchCV(
    estimator=solution.estimator,
    search_spaces=solution.search_spaces,
    n_iter=config['hyperparam_search']['n_iter'],
    scoring=config['hyperparam_search']['scoring'],
    refit=config['hyperparam_search']['scoring'][0],
    # error_score=0,
    # n_jobs=-1,
    verbose=10,
    cv=cv_hyperparams
)

results = hyperparam_search.fit(solution.X, solution.y)
results_df = pd.DataFrame(results.cv_results_)
# results_df = results_df.sort_values('rank_test_roc_auc')

# save results

timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
results_path = os.path.join(config['output_dir'], f'training_{timestamp}')
os.makedirs(results_path, exist_ok=True)
copied_config_path = os.path.join(results_path, 'model_selection.yaml')
with open(copied_config_path, 'w') as copied_config_file:
    yaml.dump(config, copied_config_file)

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

search_cv_path = os.path.join(results_path, 'search_cv2.pickle')
clone = copy(hyperparam_search)
# remove KerasClassifier wrapper instance
# (there is a known bug if we try to persist it with pickle)
clone.best_estimator_ = None
with open(search_cv_path, 'wb') as out:
    pickle.dump(clone, out)
