import pandas as pd
from training_model_sel import hyperparam_search, X, y, RANDOM_STATE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate

# %%
# estimate the OOS performance for the best model
# cross-validates (outer CV) the inner CV procedure

cv_oos_performance = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=RANDOM_STATE
)

oos_performance = cross_validate(
    estimator=hyperparam_search,
    X=X,
    y=y,
    cv=cv_oos_performance,
    return_estimator=True
)
print('expected oos performance', oos_performance)
pd.DataFrame(oos_performance)
