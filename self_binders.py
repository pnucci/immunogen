#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
df = pd.read_excel(
    'raw/jci-126-88590-s002.xlsx',
    engine='openpyxl',
    sheet_name='Supplemental Table 2',
    skiprows=[0]
    )
#%%
# df.to_csv('self_binders.csv')
#%%
df = df[
    (df['Allele']=='HLA-A*02-01') &
    (df['Length']==9)
    ]
    
peptides = list(df['Peptide Sequence'].unique())
print(len(peptides))
# print(peptides)
