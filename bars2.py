# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:31:48 2020

@author: qtckp
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

y = np.random.rand(10,4)
y[:,0]= np.arange(10)
df = pd.DataFrame(y, columns=["X", "A", "B", "C"])

df['B'] = df['B'] + df['A']

ax = df.plot(x="X", y="B", kind="bar")
df.plot(x="X", y="A", kind="bar", ax=ax, color="C2")


plt.show()




