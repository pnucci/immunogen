# %%
import pandas as pd
import numpy as np
from tensorflow.keras import utils
from original_approach.dataset import pathogens, self_binders
from original_approach.encoding import encode_similarity_matrix


# %%
X_seqs = pathogens.peptides + self_binders.peptides
X = encode_similarity_matrix(X_seqs)
# %%
y = np.array(
    [1]*len(pathogens.peptides)
    +
    [0]*len(self_binders.peptides)
)
y_cat = utils.to_categorical(y)

# %%

num_outputs = len(np.unique(y))
