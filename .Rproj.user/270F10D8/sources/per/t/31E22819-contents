
library(tidyverse)
library(magrittr)


# просто считываю данные и удаляю лишние столбцы

foods = read_csv('foods.csv')

recipes = read_csv('recipes.csv')

foods %<>% select_if(is.numeric) %>% 
  select(-category_id, -general, -id_1, -food_id, -id_2, -food_id_1, -id_3, food_id_2, -food_id_2)

recipes %<>% select_if(is.numeric) %>% select(-recipe_id) 


# отбираю только общие столбцы

foods_cols = colnames(foods)

recipes_cols = colnames(recipes)

intersect(foods_cols, recipes_cols)

setdiff(recipes_cols, foods_cols)

setdiff(foods_cols, recipes_cols)



right_columns = intersect(foods_cols, recipes_cols)

foods = foods[,right_columns]

recipes = recipes[,right_columns]




# goal tabs

goal = read_csv('goal.csv')

goal_columns = colnames(goal)


setdiff( goal_columns, right_columns)

# надо убрать столбец, так как нет соответствия, и несколько переименовать (это нехорошо)
# а еще есть нулевой столбец carbohydrate и нормальный carbohydrates, который должен быть carbohydrate

goal %<>% select(-bromine, -carbohydrate) %>%  rename(
  'fat' = 'fats',
  'energy' = 'calories',
  'protein' = 'proteins',
  'carbohydrate' = 'carbohydrates',
  'chrome' = 'chromium',
  'omega_3' = 'omega3',
  'selen' = 'selenium'
  ) 


goal_columns = colnames(goal)

foods = foods[,goal_columns]

recipes = recipes[,goal_columns]



