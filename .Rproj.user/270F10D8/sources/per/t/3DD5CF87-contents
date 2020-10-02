

library(tidyverse)

mat = matrix(1, nrow = 4, ncol = length(right_columns))

rownames(mat) = c('day_bottom','day_top','week_bottom', 'week_top')
colnames(mat) = right_columns

mat = as_tibble(mat)

# a
mat[,c('energy', 'fat', 'protein', 'carbohydrate')] = c(0.93, 1.07, 0.97, 1.03)

# b
mat[,c('beta_carotene',
       'vitamin_b1',
       'vitamin_b2','vitamin_b5','vitamin_b6','vitamin_b9','vitamin_c',
       'vitamin_e','vitamin_pp','vitamin_k','vitamin_h','chlorine','magnesium',
       'sodium','phosphorus','iron','iodine','manganese',
       'copper', 'molybdenum','cobalt', 
       'selen','serine', 'chrome','zinc')] = c(0.8, 3.5, 0.95, 3.5)


# c
mat[,c('vitamin_a','vitamin_b12','calcium','chlorine','potassium')] = c(0.8, 2.5, 0.95, 2.5)


# d (-bromine)
#mat[, c('vitamin_d','fluorine','silicon','bor','vanadium','','','','','','','','',)] =

  
mat[, c('omega_6','omega_9')] = c(0.8, 1.8, 0.95, 1.1)
mat[,'omega_3'] = c(0.8, 10, 0.95, 3)

# Соотношение омега-3/омега-6 от 2/1 до 1/10
  

# тут описание расплывчатое, надо добавить еще углеводов и жиров
#mat[, c('cholesterol','sugars','sfa','purines','oxalic')] = c(0, 1.2,0,5)

mat[, c('cholesterol','fructose','galactose','glucose','saccharose')] = c(0, 1.2,0,1.5) 

  






bad_cols = sapply(mat, function(cl) sum(cl == 1)==4)  

write_csv(mat[,!bad_cols], 'borders.csv')  
  
  
#mat[,bad_cols] = c(1,100,1,100)
#write_csv(mat, 'borders.csv') 



