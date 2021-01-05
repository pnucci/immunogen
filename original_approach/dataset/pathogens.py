# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from original_approach.encoding import encode_similarity_matrix

df = pd.read_csv('raw/epitope_table_export_1608718806.csv', skiprows=[0])
df


# %%
df['len'] = df['Description'].apply(len)


peptides = list(df[df['len'] == 9]['Description'].unique())
print(len(peptides))
# print(peptides)

invalid = []
valid = []
for s in peptides:
    try:
        encode_similarity_matrix([s])
        valid += [s]
    except Exception as e:
        invalid += [s]

if len(invalid):
    print('peptides not included due to error')
    print(invalid)
peptides = valid
