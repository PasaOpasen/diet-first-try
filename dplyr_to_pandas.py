# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:55:45 2020

@author: qtckp
"""
import numpy as np
import pandas as pd


# просто считываю данные и удаляю лишние столбцы

foods = pd.read_csv('foods.csv')
foods_names = foods.food_id.astype(str)

recipes = pd.read_csv('recipes.csv')
recipes_names = recipes.id.astype(str) 


foods = foods.iloc[:,1:].dropna(axis=1, how='all').select_dtypes(include = np.number).drop('food_id', 1).fillna(0)


recipes = recipes.dropna(axis=1, how='all').select_dtypes(include = np.number).drop(['recipe_id','id','coef_for_men','coef_for_women'],1)



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

goal = pd.read_csv('goal.csv')




# надо убрать столбец, так как нет соответствия, и несколько переименовать (это нехорошо)
# а еще есть нулевой столбец carbohydrate и нормальный carbohydrates, который должен быть carbohydrate

renames = {
      'fat':'fats',
  'energy': 'calories',
  'protein': 'proteins',
  'carbohydrate': 'carbohydrates',
  'chrome': 'chromium',
  'omega_3': 'omega3',
  'selen': 'selenium'
    }

goal = goal.drop('carbohydrate',1).rename( columns = {value:key for key, value in renames.items()},
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
# если есть в цели, но не в коридоре, добавляем в коридор с границами 1-10

borders = pd.read_csv('borders.csv')

tmp = goal_columns.intersection(borders.columns)
borders = borders.loc[:,tmp]

# как я понял, это не нужно, потому что исходим только от имеющихся диапазонов, норму прям соблюдать не обязательно
#tmp = setdiff(goal_columns, tmp)
#borders[,tmp] = c(1, 10, 1, 10)

goal_columns = tmp
goal = goal.loc[:,goal_columns]
foods = foods.loc[:,goal_columns]
recipes = recipes.loc[:,goal_columns]


#foods['name'] = foods_names
recipes['name'] = recipes_names


foods.to_csv('currect_foods.csv', index = False)
recipes.to_csv('currect_recipes.csv', index = False)




borders_result = borders.iloc[0,:] * goal

for i in range(1, 4):
    borders_result = borders_result.append(borders.iloc[i,:] * goal)

borders = borders_result.append(goal)



borders.to_csv('currect_borders.csv', index = False)




indexes = {
    'recipes_names': recipes_names,
    'foods_names': foods_names,
    'goal_columns': list(borders.columns)
    }









