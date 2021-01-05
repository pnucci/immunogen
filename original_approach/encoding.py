# %%
import numpy as np

residue_similarity_order = 'GAVLISTYFWDEQNKRHCMP'

residue_indexes = {res: ix for ix, res in enumerate(residue_similarity_order)}


def encode_similarity_matrix(seqs):
    matrices = []
    for seq in seqs:
        matrix = np.zeros((len(seq), len(residue_similarity_order)))
        for row, res in enumerate(seq):
            ix = residue_indexes[res]
            matrix[row, ix] = 1
        matrices += [matrix]
    return np.stack(matrices)


# %%
print(encode_similarity_matrix(['GAV']))
