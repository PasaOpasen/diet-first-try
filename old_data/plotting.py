# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:20:58 2020

@author: qtckp
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

val = np.array([23,78,22,19,45,33,20])
res = np.random.normal(3,0.5, (val.size,))

df = pd.DataFrame({
    'name':['john','mary','peter','jeff','bill','lisa','jose'],
    'age': val,
    'upper': val + res,
    'lower': val - res
})


# a simple line plot
fig, ax = plt.subplots()

ax = df.plot(kind='bar',x='name',y='age', ax = ax)

df.plot(kind= 'line', x='name', y='upper', ax = ax, color = 'red')

df.plot(kind= 'line', x='name', y='lower', ax = ax, color = 'black')

