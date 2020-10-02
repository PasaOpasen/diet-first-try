
library(tidyverse)
library(magrittr)


# просто считываю данные и удаляю лишние столбцы

foods = read_csv('foods.csv')
foods_names = foods$name

recipes = read_csv('recipes.csv')
recipes_names = as.character(recipes$id) 

foods %<>% select_if(is.numeric) %>% 
  select(#-category_id, -general, -id_1, 
         -food_id, -X1
         #-id_2, -food_id_1, -id_3, food_id_2, -food_id_2
         )
foods[is.na(foods)] = 0


recipes %<>% select_if(is.numeric) %>% select(-recipe_id, -id, - coef_for_men, -coef_for_women) 



# отбираю только общие столбцы

foods_cols = colnames(foods)

recipes_cols = colnames(recipes)

intersect(foods_cols, recipes_cols)

setdiff(recipes_cols, foods_cols)

setdiff(foods_cols, recipes_cols)



#right_columns = intersect(foods_cols, recipes_cols)

#foods = foods[,right_columns]

#recipes = recipes[,right_columns]


right_columns = union(foods_cols, recipes_cols)

foods[,setdiff(recipes_cols, foods_cols)] = 0

recipes[,setdiff(foods_cols, recipes_cols)] = 0

# чтоб совпал порядок
foods = foods[,right_columns]

recipes = recipes[,right_columns]



# goal tabs

goal = read_csv('goal.csv')

goal_columns = colnames(goal)


setdiff(goal_columns, right_columns)

# надо убрать столбец, так как нет соответствия, и несколько переименовать (это нехорошо)
# а еще есть нулевой столбец carbohydrate и нормальный carbohydrates, который должен быть carbohydrate

goal %<>% select(#-bromine, 
                 -carbohydrate
                 ) %>%  rename(
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


write_csv(foods %>% mutate(name = foods_names), 'currect_foods.csv')
write_csv(recipes %>% mutate(name = recipes_names), 'currect_recipes.csv')


# connect goal with borders

# надо присоединить к цели коридоры
# если признак есть в коридоре, но не в цели, его отбрасываем
# если есть в цели, но не в коридоре, добавляем в коридор с границами 1-10

borders = read_csv('borders.csv')

tmp = intersect(goal_columns, colnames(borders))
borders = borders[,tmp]

# как я понял, это не нужно, потому что исходим только от имеющихся диапазонов, норму прям соблюдать не обязательно
#tmp = setdiff(goal_columns, tmp)
#borders[,tmp] = c(1, 10, 1, 10)

goal_columns = tmp
goal = goal[,goal_columns]
foods = foods[,goal_columns]
recipes = recipes[,goal_columns]
write_csv(foods %>% mutate(name = foods_names), 'currect_foods.csv')
write_csv(recipes %>% mutate(name = recipes_names), 'currect_recipes.csv')



# сделать порядок в коридорах таким же, как у цели
borders = borders[, goal_columns]


for(i in 1:4){
  borders[i,] = borders[i,] * as.numeric(goal)
}
borders[5,] = goal

#for(i in 1:ncol(borders)){
#   if(sum(borders[,i])==0){
#    borders[,i] = c(0,1000,0,1000)
#  }
#}

write_csv(borders, 'currect_borders.csv')

