# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:44:00 2020

@author: qtckp
"""


import json
import codecs


with codecs.open('norms.txt', 'r', encoding = 'utf-8-sig') as f:
    voc = json.load(f)


voc = voc['data']['user']['meta']['patient']['meta']['norms']


res = {}

for p in voc:
    for obj in p['nutrients']:
        res[obj['code']] = obj['mass']



import pandas as pd

df = pd.DataFrame(res.values(), index = res.keys())

df = df.transpose()

df.to_csv('goal.csv', index = 0)



