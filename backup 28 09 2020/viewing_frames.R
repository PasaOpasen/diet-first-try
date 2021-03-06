

library(tidyverse)

VIEW = function(name){
  
  data = read_csv(paste0('database/28 09/',name,'.csv'))
  View(data)
}



dfs = c(
  'categories_limit',
  'category',
  'food_limit',
  'food_tag',
  
  'minerals',
  'nutrient_group',
  'nutrients',
  'NutrientType',

  'recipes',
  'recipes_composition'
)



VIEW('food_tags')

