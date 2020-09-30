# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:28:36 2020

@author: qtckp
"""

import json
import codecs
import warnings

import numpy as np
import pandas as pd



def get_goal(file_name = 'norms.txt'):

    with codecs.open(file_name, 'r', encoding = 'utf-8-sig') as f:
        voc = json.load(f)
    
    
    voc = voc['data']['user']['meta']['patient']['meta']['norms']
    
    
    res = {}
    
    for p in voc:
        for obj in p['nutrients']:
            res[obj['code']] = obj['mass']
    

    
    df = pd.DataFrame(res.values(), index = res.keys())
    
    df = df.transpose()
    
    #df.to_csv('goal.csv', index = 0)
    
    return df




def get_data(file_name = 'norms.txt'):

    # просто считываю данные и удаляю лишние столбцы
    
    #foods = pd.read_csv('food.csv', sep =';')
    foods = pd.read_csv('all_foods.csv', sep =';', error_bad_lines=False).fillna(0)
    foods_names = foods.food_id.astype(int).astype(str)
    food_cat = foods['category_id'].values
    food_general = foods['general'].values
    
    recipes = pd.read_csv('recipes.csv')
    recipes_names = recipes.id.astype(str) 
    

    foods = foods.iloc[:,5:].dropna(axis=1, how='all').select_dtypes(include = np.number).drop('food_id', 1)#.fillna(0)
    
    
    recipes = recipes.dropna(axis=1, how='all').select_dtypes(include = np.number).drop(['recipe_id','id','coef_for_men','coef_for_women'],1)
    
    
    
    is_drink = np.arange(foods.shape[0])[food_cat == 25]
    #is_not_drink = np.arange(foods.shape[0])[food_cat != 25]
    is_food = np.arange(foods.shape[0])[food_general == 1]

    drinks = foods.iloc[is_drink,:]
    drinks_names = foods_names[is_drink] 

    foods = foods.iloc[is_food,:]
    foods_names = foods_names[is_food]


    water = {
        'recipes': recipes['water'].values,
        'foods': foods['water'].values/2,
        'drinks': drinks['water'].values/2
        }




    recipes_meal_time = np.zeros(recipes.shape[0])
    foods_meal_time = np.zeros(foods.shape[0])
    
    meal_time = pd.read_csv('food_tags.csv', names = ['food_id', 'name', 'name_1']).drop('name', 1)
    
    for meal, time in [('завтрак',1), ('обед',3), ('ужин',5)]:
        foods_meal_time[np.array([fid in meal_time[meal_time['name_1'] == meal]['food_id'] for fid in foods_names.astype(int)])] = time



    
    # отбираю только общие столбцы
    
    foods_cols = foods.columns
    
    recipes_cols = recipes.columns
    
    foods_cols.intersection(recipes_cols)
    
    
    
    right_columns = foods_cols.union(recipes_cols)
    
    foods.loc[:,recipes_cols.difference(foods_cols)] = 0
    
    recipes.loc[:,foods_cols.difference(recipes_cols)] = 0
    
    # чтоб совпал порядок
    foods = foods.loc[:,right_columns]
    
    recipes = recipes.loc[:,right_columns]
    
    
    # goal tabs
    
    goal = get_goal(file_name) #pd.read_csv('goal.csv')
    
    
   # goal = goal.loc[:, (goal != 0).any(axis=0)] # delete columns with all 0
    
    
    # надо убрать столбец, так как нет соответствия, и несколько переименовать (это нехорошо)
    # а еще есть нулевой столбец carbohydrate и нормальный carbohydrates, который должен быть carbohydrate
    
    renames = {
          'fat':'fats',
      'energy': 'calories',
      'protein': 'proteins',
      'carbohydrate': 'carbohydrates',
      'chrome': 'chromium',
      'omega_3': 'omega3',
      'omega_6': 'omega6',
      'omega_9': 'omega9',
      'selen': 'selenium'
        }
    #goal.drop('carbohydrate',1)
    goal =  goal.rename( columns = {value:key for key, value in renames.items()},
        inplace = False
      ) 
    
    
    goal_columns = goal.columns
    
    
    #foods = foods.loc[:,goal_columns]
    
    #recipes = recipes.loc[:,goal_columns]
    
    
    #write_csv(foods %>% mutate(name = foods_names), 'currect_foods.csv')
    #write_csv(recipes %>% mutate(name = recipes_names), 'currect_recipes.csv')
    
    
    # connect goal with borders
    
    # надо присоединить к цели коридоры
    # если признак есть в коридоре, но не в цели, его отбрасываем
    
    borders = pd.read_csv('borders.csv').drop('omega_9', 1)
    
    tmp = goal_columns.intersection(borders.columns)
    
    to_remove = [name for name in tmp if goal.loc[0,name] == 0]
    
    if to_remove:
        warnings.warn(f"WARNING.........columns {to_remove} are in borders and equal 0 in goal. They will be removed")
        tmp = pd.Index([name for name in tmp if name not in to_remove]) # если использовать Index.difference, порядок испортится
    
    
    borders = borders.loc[:,tmp]
    
    # как я понял, это не нужно, потому что исходим только от имеющихся диапазонов, норму прям соблюдать не обязательно
    #tmp = setdiff(goal_columns, tmp)
    #borders[,tmp] = c(1, 10, 1, 10)
    
    goal_columns = tmp
    goal = goal.loc[:,goal_columns]
    foods = foods.loc[:,goal_columns]
    recipes = recipes.loc[:,goal_columns]
    drinks = drinks.loc[:,goal_columns]



    
    
    #foods['name'] = foods_names
    recipes['name'] = recipes_names
    
    
    #foods.to_csv('currect_foods.csv', index = False)
    #recipes.to_csv('currect_recipes.csv', index = False)
    
    
    
    
    borders_result = borders.iloc[0,:] * goal
    
    for i in range(1, 4):
        borders_result = borders_result.append(borders.iloc[i,:] * goal)
    
    borders = borders_result.append(goal)
    
    
    
    #borders.to_csv('currect_borders.csv', index = False)
    
    
    indexes = {
        'recipes_names': list(recipes_names),
        'foods_names': list(foods_names),
        'drinks_names': list(drinks_names),
        'goal_columns': list(borders.columns),
        'drinks_columns': list(drinks.columns),
        'recipes_energy': recipes['energy'].values,
        'foods_enegry': foods['energy'].values/2,
        'meal_time':{
            'recipes': recipes_meal_time,
            'foods': foods_meal_time
            },
        'water': water
        }
    
    return foods.to_numpy()/2, recipes.iloc[:,:-1].to_numpy(), borders.to_numpy(), drinks.to_numpy()/2, indexes






