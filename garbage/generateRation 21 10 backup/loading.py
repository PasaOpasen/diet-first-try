# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:28:36 2020

@author: qtckp
"""
import os, sys
import json
import codecs
import warnings

import numpy as np
import pandas as pd

from collections import defaultdict

sys.path.append(os.path.dirname(__file__))
from db_local import queryR
from tags import get_meal_ways

# from coefficients import get_coefs_depended_on_goal



DIMAS_WORKING = True



bad_foods = {
    'животные жиры': 60,
    'соусы': 68,
    'сырым несъедобно': 39,
    'неорганические продукты': 69,
    'растительные масла': 57,
    'специи': 47,
    'водоросли':79
    }
bad_foods_set = set(bad_foods.values())



def df_to_dictionary(df):
      
    return {key:val for key, val in zip(df.iloc[:,0],df.iloc[:, 1])}

def get_bad_set(df):
    
    return set([key for key, val in zip(df.iloc[:,0],df.iloc[:, 1]) if val in bad_foods_set])



def get_DataFrame_from_db(query, engine):
    return queryR(query) if DIMAS_WORKING else pd.read_sql(query, con=engine)





better_tags = {
    
    'завтрак':{
        'завтрак': 4,
        'салаты': 45,
        'хлебобулочные': 64,
        'фрукты': 2,
        'орехи': 14,
        'сухофрукты': 40,
        'ягоды': 32,
        'яйца': 25
        },
    'первый перекус':{
        'перекус': 41
        },
    'обед':{
        'обед': 20,
        'первое блюдо': 73,
        'десерты': 42,
        'зелень': 44
        },
    'второй перекус':{
        },
    'ужин':{ 
        'ужин': 11
        }
    }

meal_sets = {
    
    'завтрак':set(list(better_tags['завтрак'].values()) + list(better_tags['первый перекус'].values())),
    'обед': set(list(better_tags['обед'].values()) + list(better_tags['второй перекус'].values())),
    'ужин': set(list(better_tags['ужин'].values()))
    }
        
        
       



def get_goal(goal_as_str = 'norms.txt'):
    """
    возвращает цель в виде DataFrame

    Parameters
    ----------
    goal_as_str : TYPE, optional
        название файла с нормами или строка с содержанием json. The default is 'norms.txt'.

    Raises
    ------
    Exception
        если аргумент не является названием существующего файла и не декодируется из json, возникает исключение.

    Returns
    -------
    df : pandas DataFrame
        строка цели в виде DataFrame.

    """
    
    if DIMAS_WORKING and os.path.isfile(goal_as_str):
        with codecs.open(goal_as_str, 'r', encoding = 'utf-8-sig') as f:
            voc = json.load(f)
            
    # else:
    #     try:
            
    #     except:
    #         raise Exception(f'текст "{goal_as_str}" в аргументе нормы не является ни именем существующего файлы, не строкой со словарём json')
    else:
        voc = goal_as_str
    
    voc = voc['norms']
    
    res = {key:val for key,val in voc.items() if key != 'water'}
    # res = {}
    
    # for p in voc:
    #     for obj in p['nutrients']:
    #         res[obj['code']] = obj['mass']
    

    
    df = pd.DataFrame(res.values(), index = res.keys())
    
    df = df.transpose()
    
    #df.to_csv('goal.csv', index = 0)
    
    return df, voc['water']



def get_data(goal_as_str = 'norms'):
    
    if not DIMAS_WORKING:
        from app import db
        engine = db.engine
    else:
        engine = None

    # просто считываю данные и удаляю лишние столбцы
    q = '''
                SELECT * FROM food 
                    inner join acids on food.id = acids.food_id
                    inner join minerals on food.id = minerals.food_id
                    inner join vitamins on food.id = vitamins.food_id 
                    where general = 1
                    ;
            '''    
    
    foods = get_DataFrame_from_db(q, engine).fillna(0)
    
    
    foods_names = foods.food_id.iloc[:, 0].astype(int).astype(str)
    food_cat = foods['category_id'].values
    #food_general = foods['general'].values
    
    
    q = '''select recipes_composition.* from recipes_composition, recipes where recipe_id = recipes.id and recipes.deleted_at is null'''
    
    recipes = get_DataFrame_from_db(q, engine)


    meal_time_f = get_DataFrame_from_db('select * from food_tag', engine)
    meal_time_r = get_DataFrame_from_db('select * from recipe_tag', engine)
    
    
    bad_food_inds = get_bad_set(meal_time_f)
    
    
    food_tags = df_to_dictionary(meal_time_f)
    food_tags = defaultdict(int, food_tags)
    
    #uniqs1 = np.unique(foods[foods['general']==1]['food_id'].astype(int).values)
    #uniqs2 = np.unique(meal_time_f.iloc[:,0].values)
    #set(uniqs1)-set(uniqs2)    


    #print(recipes.iloc[:, :3].describe())

    recipes_names = recipes.recipe_id.astype(str) 
    

    foods = foods.iloc[:,5:].dropna(axis=1, how='all').select_dtypes(include = np.number).drop('food_id', 1)#.fillna(0)
    
    recipes = recipes.dropna(axis=1, how='all').select_dtypes(include = np.number).drop(['recipe_id','id'],1)
    
    
    
    # все продукты, которые напитки, нужно отнести в отдельную таблицу
    # вдобавок из продуктов оставить только general
    
    is_drink = np.arange(foods.shape[0])[(food_cat == 25)]# | np.array([food_tags[int(name)] == 34 for name in foods_names]) ] # 34 -- напитки, раньше было 25
    
    #print(np.sum(food_cat == 25))
    #print(np.sum(np.array([food_tags[int(name)] == 34 for name in foods_names])))
    
    #is_not_drink = np.arange(foods.shape[0])[food_cat != 25]
    is_food = np.arange(foods.shape[0])[(food_cat != 25) & np.array([fc not in bad_food_inds for fc in foods_names.astype(int)])]

    print(f'removed {np.sum(((food_cat == 25) | np.array([fc in bad_food_inds for fc in foods_names.astype(int)])))} bad foods')
    
    drinks = foods.iloc[is_drink,:]
    drinks_names = foods_names[is_drink] 

    foods = foods.iloc[is_food,:]
    foods_names = foods_names[is_food]
    
    #food_cat = np.array([food_tags[name] for name in foods_names.astype(int).values])
    #print(f'{np.sum(np.array([fc in bad_foods_set for fc in food_cat]))} bad foods')

    water = {
        'recipes': recipes['water'].values,
        'foods': foods['water'].values,
        'drinks': drinks['water'].values
    }




    # значения по времени приёма пищи:
    # 1 -- завтрак, 3 -- обед, 5 -- ужин
    # что не относится к 1/3/5, будет 0
    # 0 -- это любое время
    # перед генерацией разбиения все 0 случайно заменятся на 1/3/5

    recipes_meal_time = np.zeros(recipes.shape[0])
    foods_meal_time = np.zeros(foods.shape[0])
    # здесь находим id всех, например, продуктов с тегом "завтрак" 
    # в соответствующем массиве на местах этих тегов ставим 1, потому что завтрак
    
    # tag_to_number = {
    #     'завтрак': 4,
    #     'обед': 20,
    #     'ужин': 11
    #     }
    
    def is_in_set(array, set_):
        return np.array([val in set_ for val in array])
    
    food_names_int = foods_names.astype(int)
    recipes_names_int = recipes_names.astype(int)
    for meal, time in [('обед',3), ('завтрак',1),  ('ужин',5)]: # сменил порядок, чтоб закидывание в обед было приоритетным
        food_set = set(meal_time_f[is_in_set(meal_time_f['tag_id'].values, meal_sets[meal])]['food_id'].astype(int).values) # speed up over 5-6 times!
        recipe_set = set(meal_time_r[is_in_set(meal_time_r['tag_id'].values, meal_sets[meal])]['recipe_id'].astype(int).values)
        
        #print(f"{np.sum(meal_time_r['tag_id'] == tag_to_number[meal])}")
        #print(f"{np.sum(is_in_set(meal_time_r['tag_id'].values, meal_sets[meal]))}")
        
        foods_meal_time[np.array([fid in food_set for fid in food_names_int])] = time
        recipes_meal_time[np.array([fid in recipe_set for fid in recipes_names_int])] = time

    
    # get ways 
    food_ways = {}
    food_tags = {}
    recipe_ways = {}
    recipe_tags = {}
    for obj_id in foods_names:
        tgs = meal_time_f[meal_time_f['food_id'] == int(obj_id)]['tag_id']
        food_ways[obj_id] = get_meal_ways(tgs)
        food_tags[obj_id] = set(tgs)
     
    for obj_id in recipes_names:
        tgs = meal_time_r[meal_time_r['recipe_id'] == int(obj_id)]['tag_id']
        recipe_ways[obj_id] = get_meal_ways(tgs)
        recipe_tags[obj_id] = set(tgs)

    

    
    # отбираю только общие столбцы
    
    foods_cols = foods.columns
    
    recipes_cols = recipes.columns
    
    foods_cols.intersection(recipes_cols)
    
    
    
    right_columns = foods_cols.union(recipes_cols)
    
    foods.loc[:,recipes_cols.difference(foods_cols)] = 0
    drinks.loc[:,recipes_cols.difference(foods_cols)] = 0
    
    for col in foods_cols.difference(recipes_cols):
        recipes[col] = 0
    
    # чтоб совпал порядок
    
    #foods = foods.loc[:,right_columns]
    
    #recipes = recipes.loc[:,right_columns]
    
    
    # goal tabs
    
    goal, water['needed'] = get_goal(goal_as_str) #pd.read_csv('goal.csv')
    
    
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
    

    
    # coefs = {
    #     'recipes': get_coefs_depended_on_goal(recipes, goal),
    #     'foods': get_coefs_depended_on_goal(foods/2, goal),
    #     'drinks': get_coefs_depended_on_goal(drinks/2, goal)
    #     }
    
    
    
    #foods = foods.loc[:,goal_columns]
    
    #recipes = recipes.loc[:,goal_columns]
    
    
    #write_csv(foods %>% mutate(name = foods_names), 'currect_foods.csv')
    #write_csv(recipes %>% mutate(name = recipes_names), 'currect_recipes.csv')
    
    
    # connect goal with borders
    
    # надо присоединить к цели коридоры
    # если признак есть в коридоре, но не в цели, его отбрасываем
    
    if DIMAS_WORKING:
        borders = pd.read_csv('borders.csv').drop('omega_9', 1)
        importances = pd.read_csv('nutrient_importances.csv')
    else:
        borders = pd.read_csv('app/recomendation/generateRation/borders.csv').drop('omega_9', 1)
        importances = pd.read_csv('app/recomendation/generateRation/nutrient_importances.csv')
    
    tmp = goal_columns.intersection(borders.columns).intersection(right_columns)
    
    to_remove = [name for name in tmp if goal.loc[0,name] == 0]
    
    if to_remove:
        warnings.warn(f'WARNING.........columns {to_remove} are in borders and equal 0 in goal. They will be removed')
        tmp = pd.Index([name for name in tmp if name not in to_remove]) # если использовать Index.difference, порядок испортится
    
    
    borders = borders.loc[:,tmp]
    
    # как я понял, это не нужно, потому что исходим только от имеющихся диапазонов, норму прям соблюдать не обязательно
    #tmp = setdiff(goal_columns, tmp)
    #borders[,tmp] = c(1, 10, 1, 10)
    
    goal_columns = tmp
    goal = goal.loc[:, goal_columns]
    foods = foods.loc[:, goal_columns]
    recipes = recipes.loc[:, goal_columns]
    drinks = drinks.loc[:, goal_columns]
    importances = importances.loc[:, goal_columns]

    start_dataframes = {
        'recipes': recipes.copy(),
        'foods': foods.copy(),
        'drinks':drinks.copy(),
        'goal': goal.copy(),
        'importances': importances.copy()
        }

    
    
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
        'foods_enegry': foods['energy'].values,
        'meal_time':{
            'recipes': recipes_meal_time,
            'foods': foods_meal_time
            },
        'water': water,
        'start_dataframes': start_dataframes,
        'ways': {
            'foods': food_ways,
            'recipes': recipe_ways
            },
        'tags': {
            'foods': food_tags,
            'recipes': recipe_tags
            }
        #'coefficients': coefs
        }
    
    return foods.to_numpy(), recipes.iloc[:,:-1].to_numpy(), borders.to_numpy(), drinks.to_numpy(), indexes



if __name__ == '__main__':
    
    foods, recipes, borders, drinks, indexes = get_data('norms.txt')


