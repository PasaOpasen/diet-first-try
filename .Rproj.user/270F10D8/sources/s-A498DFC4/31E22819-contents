
library(tidyverse)
library(magrittr)

foods = read_csv('foods.csv')

recipes = read_csv('recipes.csv')

foods %<>% select_if(is.numeric) %>% 
  select(-category_id, -general, -id_1, -food_id, -id_2, -food_id_1, -id_3, food_id_2, -food_id_2)

recipes %<>% select_if(is.numeric) %>% select(-recipe_id) 


foods_cols = colnames(foods)

recipes_cols = colnames(recipes)

intersect(foods_cols, recipes_cols)

setdiff(recipes_cols, foods_cols)

setdiff(foods_cols, recipes_cols)



right_columns = intersect(foods_cols, recipes_cols)

foods = foods[,right_columns]

recipes = recipes[,right_columns]











