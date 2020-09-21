# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:40:20 2020

@author: qtckp
"""

import numpy as np
import pandas as pd

from collections import defaultdict
from split_by_sums import get_split_by_sums, get_sums_by_classes



def splitDay(recipes_energy, foods_energy, recipes_vals, foods_vals, recipes_names, foods_names, recipes_classes, foods_classes, sums = [[15,10], 40, 35] , random_labels = [1,3,5], tol = 10, max_tryes = 20 ):
    
    if random_labels != None:
        recipes_classes = np.random.choice(random_labels, recipes_vals.size, replace = True)
        foods_classes = np.random.choice(random_labels, foods_vals.size, replace = True)
    
    def get_triple(energy, vector, classes, names, type_object = 'recipe'):
        """
        делает таблицу из (тип, имя, энергия, начальный класс), которые надо будет раскидывать
        
        прикол в том, что все повторы еды надо превращать в отдельные примеры, чтобы не перекидывать сразу все вместе
        """
        res = [] # ids, values, classes
        for i, val in enumerate(vector):
            if val>0:
                for _ in range(int(val)):
                    res.append([type_object, names[i], energy[i], classes[i]])
        return pd.DataFrame(res, columns = ['type', 'id', 'energy', 'class'])
    
    
    s1 = get_triple(recipes_energy, recipes_vals, recipes_classes, recipes_names, 'recipes')
    s2 = get_triple(foods_energy, foods_vals, foods_classes, foods_names, 'foods')
    
    total = pd.concat([s1,s2])
    #print(total)
    
    s = []
    for obj in sums:
        if type(obj) != type([]):
            s.append(obj)
        else:
            s.append(sum(obj))
    
    ans, lst = get_split_by_sums(total['energy'].values, total['class'].values, np.array(s)/sum(s), tol)
    
    # for _ in range(max_tryes):
    #     ans, lst = get_split_by_sums(total['energy'].values, total['class'].values, np.array(s)/sum(s), tol)
    #     if lst:
    #         break
    # else:
    #     return None
    
    total['class'] = ans


    for s, tag in zip(sums, [1, 3, 5]):
        if type(s) == type([]):
            mask = np.arange(total.shape[0])[(total['class'] == tag).values]
            tot2 = total.iloc[mask,:]
            for _ in range(max_tryes):
                ans, lst = get_split_by_sums(tot2['energy'].values, np.random.choice([tag, tag +1], tot2.shape[0], True), np.array(s)/sum(s), tol)
                if lst:
                    break
            else:
                return None
            
            total.iloc[mask, 3] = ans
    
    total['class'] = total['class'].astype(str)
    total['id'] = total['id'].astype(str) 
    
    dic = get_sums_by_classes(total['energy'], total['class'])
    dic = {key: value*100 / sum(dic.values()) for key, value in dic.items()}
    

    answer = {key:{tp:defaultdict(int) for tp in ['recipes', 'foods']} for key in np.unique(total['class'])}
    for _, row in total.iterrows():
        answer[row['class']][row['type']][row['id']] += 1
    
    answer = {key:{tp:{k:v for k, v in answer[key][tp].items()} for tp in ['recipes', 'foods']} for key in np.unique(total['class'])}
    
    for key, val in dic.items():
        answer[key]['percent_of_sum'] = val
    
    
    return answer
    
    
    
#d = candidates[0]

#splitDay(recipes[:,0], foods[:,0], d.recipes_weights, d.food_weights, indexes['recipes_names'], indexes['foods_names'], None, None, sums = [[20, 10], [35, 10], [20, 5]])    


