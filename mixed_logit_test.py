#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 02:27:34 2020

@author: das
"""

from mixedlogit import ChoiceModel
import numpy as np

import pandas as pd

df = pd.read_csv("https://www.dropbox.com/s/sd3tqwlpfywu7c6/data.csv?dl=1")
# Remove unbalanced panels (future versions could handle unbalanced panels)
count_mix_by_id = np.unique(df.person_id.values, return_counts=True)
df = df[~df.person_id.isin(count_mix_by_id[0][count_mix_by_id[1] != 45])] 

df.price = -1*df.price/10000
df.operating_cost = -1*df.operating_cost

varnames = ['high_performance','medium_performance','price', 'operating_cost', 
            'range', 'electric', 'hybrid'] 
X = df[varnames].values
y = df['choice'].values

asvars = varnames
alternatives =['car','bus','bike']
randvars = {'price': 'ln', 'operating_cost': 'ln', 'range': 'ln', 'electric':'n',
            'hybrid': 'n'}
n_draws = 100
mixby = df.person_id.values #For panel data

model = ChoiceModel()
model.fit(X, y, alternatives, varnames, asvars,  randvars, mixby)
