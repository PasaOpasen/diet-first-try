# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:40:20 2020

@author: qtckp
"""

import numpy as np
import pandas as pd

from collections import defaultdict
from split_by_sums import get_split_by_sums, get_sums_by_classes


splitter_map = {
        # '1': 'завтрак',
        # '2': 'перекус после завтрака',
        # '3': 'обед',
        # '4': 'перекус после обеда',
        # '5': 'ужин',
        # '6': 'перекус после ужина'
        '1': 'breakfast',
        '2': 'snack_after_breakfast',
        '3': 'lunch',
        '4': 'snack_after_lunch',
        '5': 'dinner',
        '6': 'snack_after_dinner'
    }

def splitDay(recipes_energy, foods_energy, recipes_vals, foods_vals, recipes_names, foods_names, recipes_classes, foods_classes, sums = [[15,10], 40, 35] , random_labels = [1,3,5], tol = 10, max_tryes = 20 ):
    """
    Разбивает все продукты и рецепты в дне так, чтобы калорийность по каждому приему пищи была близка к sums c ожибкой в tol%

    Parameters
    ----------
    recipes_energy : калорийность каждого рецепта, массив нампай
        DESCRIPTION.
    foods_energy : то же самое для еды
        DESCRIPTION.
    recipes_vals : количества каждого рецепта (вектор, который хранится в дне)
        DESCRIPTION.
    foods_vals : то же самое по еде
        DESCRIPTION.
    recipes_names : id рецептов
        DESCRIPTION.
    foods_names : id еды
        DESCRIPTION.
    recipes_classes : начальные (предпочтительные) классы рецептов. если None, заполнятся рандомно, если random_labels != None
        DESCRIPTION.
    foods_classes : то же самое по продуктам
        DESCRIPTION.
    sums : list из процентов длины 3, причём каждый элемент списка может быть списком длины 2 (что значит, например, обед разбивается на сам обед и перекус после обеда), optional
        DESCRIPTION. The default is [[15,10], 40, 35].
    random_labels : метки классов в соответствии с splitter_map, optional
        если не None, recipes_classes и foods_classes заполнятся рандомно этими метками. The default is [1,3,5].
    tol : процент разрешимой ошибки, optional
        DESCRIPTION. The default is 10.
    max_tryes : максимальное число попыток при поиске разбиения, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
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
                
                rand_split = np.random.choice([tag, tag + 1], tot2.shape[0], True)
                #print(rand_split)
                #while np.unique(rand_split).size < 2:
                #    print(rand_split)
                #    rand_split = np.random.choice([tag, tag + 1], tot2.shape[0], True)
                
                ans, lst = get_split_by_sums(tot2['energy'].values, rand_split, np.array(s)/sum(s), tol)
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
    
    #answer = {splitter_map[key]:{tp:{k:v for k, v in answer[key][tp].items()} for tp in ['recipes', 'foods']} for key in np.unique(total['class'])}
    answer = {splitter_map[key]:{tp:{k:v/coef for k, v in answer[key][tp].items()} for tp, coef in zip(['recipes', 'foods'], (1, 2))} for key in np.unique(total['class'])}
    
    for key, val in dic.items():
        answer[splitter_map[key]]['percent_of_sum'] = val
    
    
    return answer
    
    
    
#d = candidates[0]

#splitDay(recipes[:,0], foods[:,0], d.recipes_weights, d.food_weights, indexes['recipes_names'], indexes['foods_names'], None, None, sums = [[20, 10], [35, 10], [20, 5]])    


