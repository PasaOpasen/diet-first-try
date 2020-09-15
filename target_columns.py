# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:02:42 2020

@author: qtckp
"""



import json
import codecs
import pandas as pd


def get_tables(goal = 'norms.txt', borders = 'borders.csv', foods = 'currect_foods0.csv', recipes = 'currect_recipes0.csv'):

    with codecs.open(goal, 'r', encoding = 'utf-8-sig') as f:
        voc = json.load(f)
    
    
    voc = voc['data']['user']['meta']['patient']['meta']['norms']
        
    res = {}
    
    for p in voc:
        for obj in p['nutrients']:
            res[obj['code']] = obj['mass']
        
    
    
    df_goal = pd.DataFrame(res.values(), index = res.keys())
    
    df_goal = df_goal.transpose()
    
    #df_goal.to_csv('goal.csv', index = 0)
    
    table_borders = pd.read_csv(borders)
    table_foods = pd.read_csv(foods)
    table_recipes = pd.read_csv(recipes)
    
    tmp = df_goal.columns.intersection(table_borders.columns)
    
    
    table_borders = table_borders.loc(tmp)
    df_goal = df_goal.loc(tmp)
    
    
    
    








