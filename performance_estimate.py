# Estimates the OOS performance for the winner of the model selection procedure
# uses nested cross-validation (each split of the outer CV might have a different winner)

import pandas as pd
from model_selection import hyperparam_search, solution, RANDOM_STATE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate

# %%

cv_oos_performance = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=RANDOM_STATE
)

oos_performance = cross_validate(
    estimator=hyperparam_search,
    X=solution.X,
    y=solution.y,
    cv=cv_oos_performance,
    return_estimator=True
)
print('expected oos performance', oos_performance)
pd.DataFrame(oos_performance)
