from original_approach.dataset import X_seqs, y, num_outputs
from chem_props_approach.encoding import encode_chemical_properties

X = encode_chemical_properties(X_seqs)