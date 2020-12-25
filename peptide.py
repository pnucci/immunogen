# %%
import numpy as np

residue_similarity_order = 'GAVLISTYFWDEQNKRHCMP'

residue_indexes = {res: ix for ix, res in enumerate(residue_similarity_order)}


def encode_peptide_seq(seq):
    matrix = np.zeros((len(seq), len(residue_similarity_order)))
    for row, res in enumerate(seq):
        ix = residue_indexes[res]
        matrix[row,ix] = 1
    return matrix
    
#%%
print(encode_peptide_seq('GAV'))