#%%
%reload_ext autoreload
%autoreload 2
import pandas as pd
import numpy as np
from dataset import X, y
import matplotlib.pyplot as plt


#%%

avgs = pd.DataFrame({'X':X,'y':y}).groupby('y').apply(lambda group: np.stack(group['X'].values).mean(axis=0))
avgs[1].shape

# %%
plt.imshow(avgs[0], cmap='gray', vmin=0, vmax=1)
plt.show()
plt.imshow(avgs[1], cmap='gray', vmin=0, vmax=1)
# %%
