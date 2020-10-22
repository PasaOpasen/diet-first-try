# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:05:16 2020

@author: qtckp
"""

# важности из файла важностей

import pandas as pd



imp1 =[
       'energy',
       'fat',
       'protein',
       'carbohydrate'
       ] 


imp2 =[
       'vitamin_b1',
       'vitamin_b2',
       'vitamin_c',
       'magnesium',
       'phosphorus',
       'iron',
       'copper',
       'cobalt',
       'sulfur',
       'zinc',
       'omega_3',
       'fiber'
       ] 


imp3 =[
       'beta_carotene',
       'vitamin_b5',
       'vitamin_b6',
       'vitamin_b9',
       'vitamin_pp',
       'vitamin_e',
       'vitamin_k',
       'chrome',
       'vitamin_h',
       'choline',
       'manganese',
       'molybdenum',
       'omega_9'
       ] 


imp4 =[
       'vitamin_a',
       'potassium',
       'sodium',
       'chlorine',
       'omega_6'
       ] 


imp5 =[
       'vitamin_b12',
       'selen',
       'iodine',
       'calcium'
       ] 


imp6 =[
       'purines',
       'cholesterol',
       'oxalic',
       'sugars',
       'sfa'
       ] 


imp7 =[
       'vitamin_d',
       'fluorine',
       'silicon',
       'bor',
       'bromine',
       'vanadium',
       'phytosterol',
       'lutein'
       ] 


total = [
    (imp1, 10, 10),
    (imp2, 9, 5),
    (imp3, 8, 3),
    (imp4, 9, 8),
    (imp5, 7, 5),
    (imp6, 1, 9),
    (imp7, 1, 1)
    ]



tmp = []

for st, mn, mx in total:
    for item in st:
        tmp.append((item, mn, mx))
     
        
df = pd.DataFrame({
    'nutrient': pd.Series([p[0] for p in tmp]),
    'lower_coef': pd.Series([p[1] for p in tmp]),
    'upper_coef': pd.Series([p[2] for p in tmp])
    })


df.set_index('nutrient').T.to_csv('nutrient_importances.csv', index = False)




