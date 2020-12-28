# %%
# %load_ext autoreload
# %autoreload 2
import pandas as pd
import numpy as np
import peptide
import pathogens
import self_binders


# %%

seqs = pathogens.peptides+self_binders.peptides
matrixes = [peptide.encode_peptide_seq(seq) for seq in seqs]
X = matrixes
# %%

y = np.array(
    [1]*len(pathogens.peptides)
    +
    [0]*len(self_binders.peptides)
)

# %%
dataset = pd.DataFrame({'X': X, 'y': y})
