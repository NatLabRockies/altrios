from pathlib import Path
import os
import pandas as pd
import re
import numpy as np
 
directory = Path().home() /\
    'Altrios_private/altrios-private/data/trips/ZANZEFF Data - v5.1 1-27-23 ALTRIOS Confidential' /\
    'ZANZEFF Data - v5 1-27-23 ALTRIOS Confidential/'

pathlist = Path(directory).glob('*.csv')
paths=[]
os.makedirs('traction/trips', exist_ok=True)
for path in pathlist:
    paths.append(path)
    df= pd.read_csv(path)
    tractive_col=[col for col in df.columns if re.search("(?i)tractive", col)]
    small_df= df[tractive_col+['PacificTime']]
    threshold = small_df[tractive_col].sum(1).max() * .05
    test = small_df[(small_df[tractive_col] > threshold).all(axis=1)].dropna(thresh=3)
    indices = [d for _, d in test.groupby(test.index - np.arange(len(test)))]
    fname = str(os.path.split(path)[1]).replace('.csv','')
    for i,ind in enumerate(indices):
        df.loc[ind.index].to_csv('traction/trips/{}_segment_{}.csv'.format(fname,i))
    