# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:08:58 2020

@author: qtckp
"""
import warnings

import numpy as np
import pandas as pd


# разбиение по группам прям как в документе
tmp_groups = {
    
    '1': ['protein', 'omega_3', 'fiber', 'vitamin_b1', 'vitamin_b2', 'vitamin_b12', 'calcium', 'magnesium', 'sulfur', 'iron', 'iodine', 'selen', 'zinc'],
    '2': ['omega_6', 'omega_9', 'starch', 'vitamin_a', 'vitamin_b5', 'vitamin_b6', 'vitamin_b9', 'vitamin_e', 'vitamin_d', 'vitamin_pp', 'vitamin_k', 'vitamin_h', 'choline', 'potassium', 'silicon', 'sodium', 'phosphorus', 'chlorine', 'bor', 'bromine', 'vanadium', 'cobalt', 'manganese', 'copper', 'molybdenum', 'fluorine', 'chrome'],
    '3': ['sugars', 'purines', 'oxalic', 'sfa', 'cholesterol'],
    '4': ['protein', 'omega_6', 'starch']#,
    #'5': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    }


supported_columns = tmp_groups['1']+tmp_groups['2']+tmp_groups['3']
supported_columns_as_index = pd.Index(supported_columns)

groups_dictionary = { }

for val in supported_columns:
    res = tuple([int(number) for number in ('1', '2', '3', '4') if val in tmp_groups[number]])
    
    groups_dictionary[val] = res





def get_coefs_by_nutrient(values, nutrient_name):
    
    groups = groups_dictionary[nutrient_name]
    
    #answer = np.zeros(values.size)
    
    if 1 in groups:
        answer = np.min(3*values, 300)
    elif 2 in groups:
        answer = np.min(2*values, 200)
    elif 3 in groups:
        
        answer = np.empty(values.size)
        
        answer[values <= 25] = 100
        
        mask = values > 25 & values <= 92
        answer[mask] = 100 - 1.5 * (values[mask]-25)
        
        answer[values > 92 & values <= 100] = 0
        
        mask = values > 100 & values <= 150
        answer[mask] = -10 * (values[mask] - 100)
        
        mask = values > 150
        answer[mask] = -500 - 20*(values[mask]-150)
        
    
    if 4 in groups:
        
        mask = values > 115
        answer[mask] -= 10*(values[mask] - 115)
        
    elif 5 in groups:
        pass
    
    return answer


def get_coefs_for_dataframe(df):
    
    lst = [get_coefs_by_nutrient(df['col'].values, 'col') for col in df.columns]
    
    return np.sum(np.array(lst), axis = 0)/100



def get_coefs_depended_on_goal(df, goal):
    
    cols = df.columns.intersection(goal.columns).intersection(supported_columns_as_index)


    to_remove = [name for name in cols if goal.loc[0,name] == 0]
    
    if to_remove:
        warnings.warn(f"WARNING.........columns {to_remove} are in coef_df and equal 0 in goal. They will be removed")
        cols = pd.Index([name for name in cols if name not in to_remove]) # если использовать Index.difference, порядок испортится
    
    
    df2 = df.loc[:, cols]
    goal2 = goal.loc[:,cols]
    
    return get_coefs_for_dataframe(df2/goal2 * 100)
        








