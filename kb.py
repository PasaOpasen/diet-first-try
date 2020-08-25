# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 02:07:33 2020

@author: qtckp
"""
import numpy as np
import scipy.stats as sc

a = np.array([1,2,3,4,5])
b = np.array([1,2,3,5,5])
c = np.array([10,4,32,213,21])
d = np.array([1,2,3,40,5])

sc.entropy(a, b)

sc.entropy(a, c)

sc.entropy(a, d)

