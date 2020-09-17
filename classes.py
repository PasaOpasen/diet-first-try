# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:36:47 2020

@author: qtckp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


def get_dict(names, values):
    return {n: v for n, v in zip(names, values)}


class Day:
    def __init__(self, recipes_weights, food_weights, combination = 0, less_than_down = None):
        self.recipes_weights = recipes_weights
        self.food_weights = food_weights
        self.combination = combination
        self.less_than_down = less_than_down
        
    def show_info(self):
        
        ri = {i: r for i, r in enumerate(self.recipes_weights) if r != 0}
        fi = {i: f for i, f in enumerate(self.food_weights) if f != 0}
        
        print('recipes index: count')
        print(ri)
        print()
        print('food index: count')
        print(fi)
        print()
        print(f'result: {self.combination} with {self.less_than_down} lesses under lower border')
    
    def to_dictionary(self, indexes):
        
        answer = {
            'recipes':[],
            'foods':[],
            'combination': get_dict(indexes['goal_columns'],self.combination), #self.combination.tolist(),
            'lower_error': int(self.less_than_down)
                  }
        
        for i, r in enumerate(self.recipes_weights):
            if r != 0:
                answer['recipes'].append({
                    'index': indexes['recipes_names'][i],
                    'count': int(r)
                    })
                
        for i, f in enumerate(self.food_weights):
            if f != 0:
                answer['foods'].append({
                    'index': indexes['foods_names'][i],
                    'count': int(f)
                    }) 
        
        return answer
    
    def to_json(self, file_name, indexes):
        
        dictionary = self.to_dictionary(indexes)
        
        with open(file_name, "w") as write_file:
            json.dump(dictionary, write_file, indent = 4)
            
    def plot(self, file_name, indexes, borders):
        
        df = pd.DataFrame({
            'nutrients': indexes['goal_columns'],
            'current result': self.combination/borders[1,:]*100,
            'lower border': borders[0,:]/borders[1,:]*100,
            'upper border': borders[1,:]/borders[1,:]*100
            })
        
        fig, ax = plt.subplots()
        

        
        
        df.plot(kind= 'line', x='nutrients', y='upper border', ax = ax, color = 'red')
        
        df.plot(kind= 'line', x='nutrients', y='lower border', ax = ax, color = 'black')
        
        df.plot(kind='bar',x='nutrients', y='current result', ax = ax)
        
        ax.set_xticklabels(df['nutrients'], rotation=90)
        
        plt.savefig(file_name, dpi = 350, bbox_inches = "tight")
        
        plt.close()

                
        
        


class Weeks:
    def __init__(self, days, configs, score, lower, upper):
        self.days = days
        self.configurations = configs
        self.score = score
        self.lower = lower
        self.upper = upper
        
    def show_info(self):
        print(f'I have {len(self.days)} days')
        print(f'My combination: {self.configurations[0]}')
        print(f'Vector: {self.configurations[1]}')
        print(f'score = {self.score}')
        print(f'lower error is {self.lower}, upper error is {self.upper}')
        
    def to_dictionary(self, indexes):
        answer = {
            
            'vector_of_combination': get_dict(indexes['goal_columns'], self.configurations[1]), #self.configurations[1].tolist(),
            'score': self.score,
            'lower_error': int(self.lower),
            'upper_error': int(self.upper),
            'days_in_week': [],
                  }
        
        for day , count in zip(self.days, self.configurations[0]):
            answer['days_in_week'].append({
                'day': day.to_dictionary(indexes),
                'repeats_in_week': int(count)
                })
        
        return answer
    
    def to_json(self, file_name, indexes):
        dictionary = self.to_dictionary(indexes)
        
        with open(file_name, "w") as write_file:
            json.dump(dictionary, write_file, indent = 4)
        








