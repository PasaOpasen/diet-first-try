# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:51:19 2020

@author: qtckp
"""

import random
from collections import namedtuple

import numpy as np
import pandas as pd

from loading import get_data
from tags import tags
from split_day import splitter_map
from classes import Day, Weeks



INNER_COEF = 0.01 # коэффициент штрафа за отлонение от нормы внутри коридора
OUTER_COEF = 1 # коэффициент штрафа за отклонение от нормы вне коридора



steps = {
    'bread': np.arange(30, 100+1, 10) / 100, # на 100 надо делить, потому что в таблице норм 1 объект рассчитан на 100 грамм
    'recipes': np.arange(150, 500+1, 10) / 100,
    'foods': np.arange(50, 200+1, 10) / 100
    }

def convert_to_scale(value, array):
    """
    выбирает из массива array ближайший элемент,
    соответствующий значению value от размаха массива
    если value = 0.2, а массив [1,2,3,4], то вернётся 1, так как 2 охватывает область 1/3 +- 1/6
    """
    return array[int(round(value*(array.size - 1)))]



f_tag = tags['первое на обед']
s_tag = tags['второе на обед']
p_tag = tags['паштет']
h_tag = tags['хлеб']

l_tag = tags['рецепты для ужина']
c_tag = tags['салат']


foods, recipes, borders, drinks, indexes = get_data('norms.txt')


np.random.seed(5)

class Bee:
    
    def __init__(self, df, inds):
        
        self.df = df
        self.df['index'] = np.arange(self.df.shape[0])
        self.indexes = inds
        self.evaluate = None
    
    def is_valid_meal_time(self, number):
        reason = p_tag if number == 2 else f_tag
        obt = lambda n: 'foods' if n == 1 else 'recipes'
        
        df = self.df[self.df['meal_time'] == number]
        
        is_reason = np.array([reason in indexes['tags'][obt(n)][ind] for ind, n in zip(df.names, df.obtains) ])
        is_result = np.array([h_tag in indexes['tags'][obt(n)][ind] for ind, n in zip(df.names, df.obtains) ])
        
        return np.logical_not(is_reason).any() or is_result.any()
        
    
    def can_be_valid(self):
        
        return self.is_valid_meal_time(2) and self.is_valid_meal_time(3)
        

    
    def get_sample(self):
        
        obt = lambda n: 'foods' if n == 1 else 'recipes'
        
        def choice(seq, count):
            if len(seq) == 0:
                return []
            return list(np.random.choice(seq, count))
        
        def contains_tag(tg):
            return np.array([tg in indexes['tags'][obt(n)][ind] for ind, n in zip(self.df.names, self.df.obtains) ])
        
        
        configs = [
            (1, 1, 2), # это значит, что на завтрак (1) берется только 1 рецепт и только 2 продукта
            (2, 1, 2),
            (4, 1, 0)
            ]
        
        result = []
        index = self.df['index']
        
        # это для простых приёмов пищи
        for meal, rec, foo in configs:
            
            ind = index[(self.df['obtains'] == 0) & (self.df['meal_time'] == meal)]
            
            result.extend(choice(ind, rec))
            
            
            if foo > 0:
            
                ind = index[(self.df['obtains'] == 1) & (self.df['meal_time'] == meal)]
            
                result.extend(choice(ind, foo))
        
        
        # для обеда и ужина есть прикольчики
        
        # первое блюдо
        ind = index[(self.df['obtains'] == 0) & (self.df['meal_time'] == 3) & contains_tag(f_tag)]
        result.extend(choice(ind, 1))
        
        self.df['is_bread'] = contains_tag(h_tag)
        if len(ind) > 0:
            # хлеб
            ind = index[(self.df['meal_time'] == 3) & self.df['is_bread'] ]
            result.extend(choice(ind, 1))
        
        # второе блюдо
        ind = index[(self.df['obtains'] == 0) & (self.df['meal_time'] == 3) & contains_tag(s_tag)]
        result.extend(choice(ind, 1))

        
        
        for meal in 5, 6:
            # рецепт для ужина
            ind = index[(self.df['obtains'] == 0) & (self.df['meal_time'] == meal) & contains_tag(l_tag)]
            result.extend(choice(ind, 1))
            # салат
            ind = index[(self.df['obtains'] == 0) & (self.df['meal_time'] == meal) & contains_tag(c_tag)]
            result.extend(choice(ind, 1))
           
        
        return Bee(self.df.iloc[np.array(result),:].sort_values('obtains').drop_duplicates('rows'), self.indexes)
    
    
    def set_evaluators(self, foods, recipes, borders, indexes):
        """
        создает внутри класса 3 оценщика для взвешенной суммы строк self.df
        """
        # выбираю строки, соответствующие моим строкам в семпле
        new_foods = foods[self.df.query('obtains == 1')['rows'],:]
        new_recipes = recipes[self.df.query('obtains == 0')['rows'],:]
        
        matrix = np.concatenate([new_recipes, new_foods])
        
        tg = borders[4, :]
        new_borders = borders[:2, :].copy()
        new_borders = (new_borders - tg)/tg # (x-goal)/goal (center+scale)
        
        importances = indexes['start_dataframes']['importances']
        
        def vector_converter(vector):
            cp = vector.copy()
            cp[vector > 1] = 1
            cp[vector < 0] = 0
            
            cp2 = np.empty_like(cp)
            
            for mask, array in zip((self.df['obtains'] == 0, self.df['obtains'] == 1, self.df['is_bread']), (steps['recipes'], steps['foods'], steps['bread'])):
                cp2[mask.values] = np.array([convert_to_scale(val, array) for val in cp[mask.values]]) 
            
            return cp2
        
        def vector_to_norms(vector):
            return (vector_converter(vector))[np.newaxis, :].dot(matrix).flatten()
            
        
        def eval_nutrients(norms):
            norms2 = (norms - tg) / tg
            total = 0
            
            less_mask = (norms2 < new_borders[0, :]).flatten() # по каким столбцам превышена нижняя граница
            high_mask = (norms2 > new_borders[1, :]).flatten() # по каким превышена верхняя граница
            okay_mask = np.logical_not(less_mask | high_mask)
            
            total += np.sum(np.abs(norms2[okay_mask]) * INNER_COEF)
            total += np.sum(np.abs(norms2[less_mask]) * importances.iloc[0, less_mask] * OUTER_COEF)
            total += np.sum(np.abs(norms2[high_mask]) * importances.iloc[1, high_mask] * OUTER_COEF)
            
            return total
            
        def eval_coef(norms):
            pass
        
        def eval_split():
            pass
        
        
        def total_eval(vector):
            norms = vector_to_norms(vector)
            
            return eval_nutrients(norms)
        
        

        self.vec2counts = vector_converter
        self.vec2norms = vector_to_norms
        self.evaluate = total_eval
        
    
    
    def get_solution(self, foods, recipes, borders, indexes):
        
        if self.evaluate == None:
            self.set_evaluators(foods, recipes, borders, indexes)
        
        rd = np.random.random(12)
        print(rd)
        ns = self.vec2norms(rd)
        print(self.evaluate(rd))
        
        rd = self.vec2counts(rd)
        print(rd)
        
        recipes_weights = np.zeros(recipes.shape[0])
        foods_weights = np.zeros(foods.shape[0])
        
        m1 = (self.df['obtains'] == 0)
        recipes_weights[self.df[m1]['rows'].values] = rd[m1.values]
        
        m2 = (self.df['obtains'] == 1)
        foods_weights[self.df[m2]['rows'].values] = rd[m2.values]
        
        
        split = {}
        df = self.df.copy()
        df['meal_time'] = df['meal_time'].astype(str)
        for ind in df['meal_time']:
            split[splitter_map[ind]] = {
                'recipes': {},
                'foods': {}
                }
        
        for name, meal_time, value, where in zip(df['names'], df['meal_time'], rd, df['obtains']):
            t = 'foods' if where == 1 else 'recipes'
            split[splitter_map[meal_time]][t][name] = value
        
        
        return Day(recipes_weights, foods_weights, ns, split)
    
    
        
                


def get_random_example(indexes):
    
    r_values = [(random.choice(indexes['ways']['recipes'][name]), i, name, 0) for i, name in enumerate(indexes['recipes_names']) if len(indexes['ways']['recipes'][name]) > 1]
    
    f_values = [(random.choice(indexes['ways']['foods'][name]), i, name, 1) for i, name in enumerate(indexes['foods_names']) if len(indexes['ways']['foods'][name]) > 1]
    
    
    r_values = [r for r in r_values if r[0] != 0]
    f_values = [f for f in f_values if f[0] != 0]
    
    sm = r_values + f_values
    
    def to_arr(ind):
        return pd.Series(np.array([r[ind] for r in sm]))
    
    return Bee(pd.DataFrame({'meal_time': to_arr(0), 'rows': to_arr(1), 'names': to_arr(2), 'obtains': to_arr(3)}), indexes)



p = get_random_example(indexes)

p.can_be_valid()

r = p.get_sample()

day = r.get_solution(foods, recipes, borders, indexes)






