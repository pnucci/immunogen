# %%
import pandas as pd
import numpy as np
from dataset import pathogens
from dataset import self_binders
from tensorflow.keras import utils
import peptide



# %%
X_seqs = pathogens.peptides + self_binders.peptides
X = peptide.encode_similarity_matrix(X_seqs)
# %%
y = np.array(
    [1]*len(pathogens.peptides)
    +
    [0]*len(self_binders.peptides)
)
y_cat = utils.to_categorical(y)

# %%

num_outputs = len(np.unique(y))
