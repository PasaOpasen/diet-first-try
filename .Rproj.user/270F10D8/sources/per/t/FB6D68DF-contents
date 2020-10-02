
library(tidyverse)

fd = read_csv(paste0('database/','food','.csv'))

fd = fd[fd$category_id == 25,]


fd = fd %>% select_if(is.numeric) %>% arrange(energy) 

write_csv(fd[,2:20], 'drinks.csv')


