# %%
import pandas as pd
import numpy as np
import peptide
import pathogens
import self_binders
from tensorflow.keras import utils


# %%
X_seqs = pathogens.peptides+self_binders.peptides
X = peptide.encode_peptide_seqs(X_seqs)

# %%
y = np.array(
    [1]*len(pathogens.peptides)
    +
    [0]*len(self_binders.peptides)
)
y_cat = utils.to_categorical(y)

# %%

num_outputs = len(np.unique(y))
