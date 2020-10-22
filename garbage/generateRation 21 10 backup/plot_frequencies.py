# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:45:54 2020

@author: qtckp


создаёт и рисует матрицы частотностей появления успешного дня для разных гиперпараметров:
    допустимое число рецептов, допустимое число повторов рецептов и попытки получения лучшего результата по еде
    
    из всех осмысленных комбинаций лучше использовать те, которые дают наибольшую частостность (тогда время вычислений будет меньше)
"""


import numpy as np

from loading import get_data
from method_draft import get_optimal_candidates



foods, recipes, borders, indexes = get_data()


recipes_count = np.arange(2, 11)
max_count = np.arange(1, 5)
tryes = np.arange(2, 17, 2)
R = np.empty((recipes_count.size, max_count.size, tryes.size))

much = 400
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        for k in range(R.shape[2]):
            R[i, j, k] = len(get_optimal_candidates(foods, recipes, borders, recipes_count[i], max_count[j], tryes[k], much, 4))/much
    print(R[i,:,:])



# import seaborn as sns
# import matplotlib.pylab as plt

for i in range(len(recipes_count)):
    # ax = sns.heatmap(R[i,:,:], linewidth=0.5, vmin=R.min(), vmax=R.max(), annot = True, cmap = 'plasma')
    # ax.set_xticklabels(tryes)
    # ax.set_yticklabels(max_count)
    # plt.xlabel('Count of food attemps')
    # plt.ylabel('Maximum count of each recipe')
    # plt.title(f'Probs for {recipes_count[i]} recipes')
    # plt.savefig(f'./day_config_probs/recipes_count = {recipes_count[i]}.png', dpi = 300)
    # plt.close()
    #plt.show()




