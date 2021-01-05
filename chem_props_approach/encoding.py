import numpy as np
from chem_props_approach import aaindex

aaindex.init(path='tmp')
residue_chem_properties = {
    'volume': aaindex.get('KRIW790103'),
    'molecular_weight': aaindex.get('FASG760101'),
    'polariz': aaindex.get('CHAM820101'),
    'conf param': aaindex.get('BEGF750103'),
    'membrane pref': aaindex.get('DESM900102'),
    'dir hyd mom': aaindex.get('EISD860103'),
    'hyd bond donors': aaindex.get('FAUJ880109'),
    'heat': aaindex.get('HUTJ700101'),
    'freq alpha helix': aaindex.get('ISOY800101'),
    'accessible mol frac': aaindex.get('JANJ790101'),
    'flexiblity param': aaindex.get('KARP850101'),
    'flexiblity param2': aaindex.get('KARP850102'),
    'side chain angle': aaindex.get('LEVM760104')
    
}
def encode_chemical_properties(seqs):
    matrices = []
    for seq in seqs:
        matrix = np.zeros((len(seq), len(residue_chem_properties)))
        for row, res in enumerate(seq):
            for col, prop in enumerate(residue_chem_properties):
                matrix[row, col] = residue_chem_properties[prop].get(res)
        matrices += [matrix]
    return np.stack(matrices)

print(encode_chemical_properties(['GAV']))