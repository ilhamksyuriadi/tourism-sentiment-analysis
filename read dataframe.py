# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:59:25 2019

@author: HP
"""

import pandas as pd
from ast import literal_eval

data = pd.read_csv("clean_dataframe.csv")
objek = pd.Series.tolist(data['objek'])
rawUlasan = pd.Series.tolist(data['ulasan'])
ulasan = []
for u in rawUlasan:
    ulasan.append(literal_eval(u))
print(type(ulasan))