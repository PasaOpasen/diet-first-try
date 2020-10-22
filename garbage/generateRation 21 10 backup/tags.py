# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:05:08 2020

@author: qtckp
"""

import pandas as pd
from db_local import queryR

tags = [
        'рецепты для завтрака',
        'мясо',
        'рыба',
        'морепродукты',
        'продукты для завтрака',
        'кофе',
        'вода',
        'соки',
        'чай',
        'рецепты для перекуса',
        'паштет',
        'хлеб',
        'продукты для завтрака',
        'первое на обед',
        'второе на обед',
        'газированные напитки',
        'салат',
        'фрукты',
        'сладкие десерты',
        'рецепты для ужина',
        'кисломолочные напитки'        
        ]


tags_df = queryR('select * from tags')

tags_dictionary = {name:id_ for id_, name in zip(tags_df.id, tags_df.name)}


tags = {key: tags_dictionary[key] for key in tags}


def is_correct(st, true, false = []):
    """
    проверяет, есть ли какой-то тег из true-названий во множестве st, при этом чтоб st не содержало теги из false-названий
    """
    
    for f in false:
        if tags[f] in st:
            return False
    
    for f in true:
        if tags[f] in st:
            return True
      
    return False


def get_meal_ways(product_tags):
    """
    выдаёт допустимые приёмы пищи для продукта с таким числом тегов
    
    0 - никакой
    1 - завтрак
    2 - перекус после завтрака
    и т. д.
    """
    
    st = set(product_tags)
    
    if is_correct(st, ['рецепты для завтрака'], ['рыба', 'мясо', 'морепродукты']):
        return [0, 1] # это значит, что рецепты для завтрака без рыбы/мяса/морепродуктов идут либо никуда (0), либо на завтрак (1)
    
    if is_correct(st, ['продукты для завтрака']):
        return [0, 1, 2]
    
    if is_correct(st, ['рецепты для перекуса'], ['мясо']):
        return [0, 2]
    
    if is_correct(st, ['первое на обед', 'второе на обед']):
        return [0, 3]
    
    if is_correct(st, ['салат'], ['мясо', 'фрукты']):
        res = [0, 4]
        if is_correct(st, ['салат'], ['мясо', 'фрукты', 'рыба', 'морепродукты']):
            res += [5, 6]
        return res
    
    if is_correct(st, ['сладкие десерты']):
        return [0, 4]
    
    if is_correct(st, ['рецепты для ужина'], ['мясо']):
        return [0, 5, 6]
    
    return [0]
    




