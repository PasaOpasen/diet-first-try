# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:17:26 2020

@author: qtckp
"""

import numpy as np
from weeksum import get7sum
from joblib import Parallel, delayed

np.set_printoptions(precision=3, suppress = True)




class Day:
    def __init__(self, recipes_weights, food_weights, combination = 0, less_than_down = None):
        self.recipes_weights = recipes_weights
        self.food_weights = food_weights
        self.combination = combination
        self.less_than_down = less_than_down


class Weeks:
    def __init__(self, days, configs):
        self.days = days
        self.configurations = configs





def currect_diff(borders, sample):
    """
    сдвигаем коридор вниз на величину образца, причем для нижнего коридора не бывает отрицательных значений
    """  
    res = borders - sample
    res[0,:] = np.maximum(0, res[0,:])
    return res


def is_valid_diff(difference):
    """
    если верхняя граница не пересечена, всё пока валидно
    """
    return  np.sum(difference[1,:] < 0) == 0  # all((v>=0 for v in difference[2,:]))


def will_be_valid_diff(borders, sample):
    
    for i in range(borders.shape[1]):
        if borders[1,i] < sample[i]:
            return False
    
    return True
    

def is_between(sample, borders):
    
    if np.sum(sample < borders[0,:]) != 0:
        return False
    
    if np.sum(sample > borders[1,:]) != 0:
        return False
    
    return True


def AE(sample, target, maxes):
    return np.sum(np.abs(sample-target)/maxes)    


def get_day_fullrandom(foods, recipes, borders, recipes_samples = 10, max_count = 3, tryes = 10):
    
    # recipes part
    
    recipes_inds = np.random.choice(recipes.shape[0], recipes_samples, replace = False)
    
    recipes_used = recipes[recipes_inds,:]
        
    counts = np.zeros(recipes_samples)
    
    bord = borders[0:2,:].copy()

    for _ in range(max_count):
        
        no_progress = 0
        
        for i in range(recipes_samples):
            
            new_bord = currect_diff(bord, recipes_used[i,:])

            if is_valid_diff(new_bord):
                bord = new_bord
                counts[i] += 1
            else:
                no_progress += 1
        
        if no_progress == recipes_samples:
            break
    
    #r = np.sum(recipes_used * counts.reshape(recipes_used.shape[0], 1), axis = 0)
    #print(np.sum(r > borders[1,:]) == 0)
    
    
    # foods part
    
    food_size = foods.shape[0]
    
    food_inds = np.arange(food_size)
    
    minval = float('inf')
    best_count2 = None
    stab = bord.copy()

    for _ in range(tryes):
        np.random.shuffle(food_inds)
        
        counts2 = np.zeros(food_size)
        bord = stab.copy()
        progress = False
        
        for i in range(food_size):
            
            while True:
                new_bord = currect_diff(bord, foods[food_inds[i],:])
                
                if is_valid_diff(new_bord):
                    bord = new_bord
                    counts2[i] += 1
                    progress = True
                else:
                    break
                
        val = np.sum(bord[0,:]/borders[0, :])
        if val < minval:
            best_count2 = np.zeros(food_size)
            best_count2[food_inds] = counts2.copy()
            minval = val
        
        if not progress:
            break
        
    
    counts2 = best_count2
    
    
    # currect weights
            
    recipes_weights = np.zeros(recipes.shape[0])
    recipes_weights[recipes_inds] = counts
    #print(recipes_weights)
    
    food_weights = best_count2
    #food_weights = np.zeros(food_size)
    #food_weights[food_inds] = counts2
    #print(food_weights)
    
    # results
    
    r = np.sum(recipes * recipes_weights.reshape(recipes.shape[0], 1), axis = 0)
    f = np.sum(foods * food_weights.reshape(food_size, 1), axis = 0)
    
    score = r + f
    #assert(np.sum(score > borders[1,:]) == 0)
    
    return Day(recipes_weights, food_weights, score, np.sum(score < borders[0,:]))


def get_day_fullrandom2(foods, recipes, borders, mins, recipes_samples = 10, max_count = 3, tryes = 10):
    
    # recipes part
    
    recipes_inds = np.random.choice(recipes.shape[0], recipes_samples, replace = False)
    
    recipes_used = recipes[recipes_inds,:]
        
    counts = np.zeros(recipes_samples)
    
    bord = borders[0:2,:].copy()

    for _ in range(max_count):
        
        no_progress = 0
        
        for i in range(recipes_samples):
            
            new_bord = currect_diff(bord, recipes_used[i,:])

            if is_valid_diff(new_bord):
                bord = new_bord
                counts[i] += 1
            else:
                no_progress += 1
        
        if no_progress == recipes_samples:
            break
    
    #r = np.sum(recipes_used * counts.reshape(recipes_used.shape[0], 1), axis = 0)
    #print(np.sum(r > borders[1,:]) == 0)
    
    
    # foods part
    
    if np.sum(mins > bord[1,:]):
        count2 = np.zeros(foods.shape[0])
        f = np.zeros(foods.shape[1])
    else:
    
        food_size = foods.shape[0]
        
        food_inds = np.arange(food_size)
        
        minval = float('inf')
        best_count2 = None
        stab = bord.copy()
    
        for _ in range(tryes):
            np.random.shuffle(food_inds)
            
            counts2 = np.zeros(food_size)
            bord = stab.copy()
            progress = False
            
            for i in range(food_size):
                
                while True:
                    new_bord = currect_diff(bord, foods[food_inds[i],:])
                    
                    if is_valid_diff(new_bord):
                        bord = new_bord
                        counts2[i] += 1
                        progress = True
                    else:
                        break
                    
            val = np.sum(bord[0,:]/borders[0, :])
            if val < minval:
                best_count2 = np.zeros(food_size)
                best_count2[food_inds] = counts2.copy()
                minval = val
            
            if not progress:
                break
            
        
        counts2 = best_count2
        
        food_weights = best_count2
        f = np.sum(foods * food_weights.reshape(food_size, 1), axis = 0)
    
    
    # currect weights
            
    recipes_weights = np.zeros(recipes.shape[0])
    recipes_weights[recipes_inds] = counts
    #print(recipes_weights)
    
    
    r = np.sum(recipes * recipes_weights.reshape(recipes.shape[0], 1), axis = 0)
    
    
    score = r + f
    #assert(np.sum(score > borders[1,:]) == 0)
    
    return Day(recipes_weights, food_weights, score, np.sum(score < borders[0,:]))


def get_day_fullrandom3(foods, recipes, borders, recipes_samples = 10, max_count = 3, tryes = 10):
    
    # recipes part
    
    recipes_inds = np.random.choice(recipes.shape[0], recipes_samples, replace = False)
    
    recipes_used = recipes[recipes_inds,:]
        
    counts = np.zeros(recipes_samples)
    
    bord = borders[0:2,:].copy()

    valid_flags = np.ones(recipes_samples, dtype = np.bool)
    for _ in range(max_count):
        
        no_progress = 0
        
        for i in range(recipes_samples):
            
            if valid_flags[i]:
            
                new_bord = currect_diff(bord, recipes_used[i,:])
    
                if is_valid_diff(new_bord):
                    bord = new_bord
                    counts[i] += 1
                else:
                    valid_flags[i] = False
                    no_progress += 1
            else:
                no_progress += 1
        
        if no_progress == recipes_samples:
            break
    
    #r = np.sum(recipes_used * counts.reshape(recipes_used.shape[0], 1), axis = 0)
    #print(np.sum(r > borders[1,:]) == 0)
    
    
    # foods part
    
    prob_food_inds = np.array([i for i, food in enumerate(foods) if np.sum(food>bord[1,:]) == 0 ])
    #print(prob_food_inds)
    print(f'{prob_food_inds.size} <= {foods.shape[0]}')
    
    if prob_food_inds.size == 0:
        food_weights = np.zeros(foods.shape[0])
        f = np.zeros(foods.shape[1])
    else:
    
        food_size = prob_food_inds.size
        
        food_inds = prob_food_inds.copy() #np.arange(food_size)
        
        minval = float('inf')
        best_count2 = None
        stab = bord.copy()
    
        for _ in range(tryes):
            np.random.shuffle(food_inds)
            
            counts2 = np.zeros(food_size)
            bord = stab.copy()
            progress = False
            
            for i in range(food_size):
                
                while True:
                    new_bord = currect_diff(bord, foods[food_inds[i],:])
                    
                    if is_valid_diff(new_bord):
                        bord = new_bord
                        counts2[i] += 1
                        progress = True
                    else:
                        break
                    
            val = np.sum(bord[0,:]/borders[0, :])
            if val < minval:
                best_count2 = np.zeros(foods.shape[0])
                best_count2[food_inds] = counts2.copy()
                minval = val
            
            if not progress:
                break
            
        
        food_weights = best_count2
        f = np.sum(foods * food_weights.reshape(food_weights.size, 1), axis = 0)
    
    
    
    
    # currect weights
            
    recipes_weights = np.zeros(recipes.shape[0])
    recipes_weights[recipes_inds] = counts
    #print(recipes_weights)
    
    
    r = np.sum(recipes * recipes_weights.reshape(recipes.shape[0], 1), axis = 0)
    
    
    score = r + f
    #assert(np.sum(score > borders[1,:]) == 0)
    
    return Day(recipes_weights, food_weights, score, np.sum(score < borders[0,:]))


def get_candidates(foods, recipes, borders, recipes_samples = 4, max_count = 3, tryes = 10, count = 100):
    #return [get_day_fullrandom(foods, recipes, borders, recipes_samples, max_count, tryes) for _ in range(count)]
    return Parallel(n_jobs=6)(delayed(get_day_fullrandom3)(foods, recipes, borders, recipes_samples, max_count, tryes) for _ in range(count))

def get_optimal_candidates(foods, recipes, borders, recipes_samples = 4, max_count = 3, tryes = 10, count = 100, max_error_count = 3):
    cands = get_candidates(foods, recipes, borders, recipes_samples, max_count, tryes, count)
    return [cand for cand in cands if cand.less_than_down <= max_error_count]



np.random.seed(5)


import pandas as pd

foods = pd.read_csv('currect_foods.csv')

#food_names = foods.name
#foods = foods.iloc[:,:-1].to_numpy()

foods = foods.to_numpy()

recipes = pd.read_csv('currect_recipes.csv')

recipes_names = recipes.name
recipes = recipes.iloc[:,:-1].to_numpy()


borders = pd.read_csv('currect_borders.csv').to_numpy()

for _ in range(20):
    d = get_day_fullrandom3(foods, recipes, borders, 4, 3, 10)
    print(d.less_than_down)
    
    
candidates = get_optimal_candidates(foods, recipes, borders, 4, 3, 10, 100, 3)


# recipes_count = np.arange(2, 11)
# max_count = np.arange(1, 5)
# tryes = np.arange(2, 17, 2)
# R = np.empty((recipes_count.size, max_count.size, tryes.size))

# much = 400
# for i in range(R.shape[0]):
#     for j in range(R.shape[1]):
#         for k in range(R.shape[2]):
#             R[i, j, k] = len(get_optimal_candidates(foods, recipes, borders, recipes_count[i], max_count[j], tryes[k], much, 3))/much
#     print(R[i,:,:])



# import seaborn as sns
# import matplotlib.pylab as plt

# for i in range(len(recipes_count)):
#     ax = sns.heatmap(R[i,:,:], linewidth=0.5, vmin=R.min(), vmax=R.max(), annot = True, cmap = 'plasma')
#     ax.set_xticklabels(tryes)
#     ax.set_yticklabels(max_count)
#     plt.xlabel('Count of food attemps')
#     plt.ylabel('Maximum count of each recipe')
#     plt.title(f'Probs for {recipes_count[i]} recipes')
#     plt.savefig(f'./day_config_probs/recipes_count = {recipes_count[i]}.png', dpi = 300)
#     plt.close()
#     #plt.show()






limit = 7


samples = [res.combination for res in candidates]

glob_borders = borders[2:4, :]

avg = borders[4,:] #np.mean(glob_borders, axis = 0)
#score = lambda sample: AE(sample, avg, glob_borders[1,:])
score = lambda sample: AE(sample, avg, avg)

weeks = get7sum(limit)
weeks[1] = [[7]]
up_lim = max(weeks.keys())

def coef_sum(inds):
    t = len(inds)
    res = []
    good = False
    for arr in weeks[t]:
        sm = samples[inds[0]]*arr[0]
        for i in range(1, len(arr)):
            sm += samples[inds[i]]*arr[i]
        sm /= 7
        res.append((arr, sm))
        if is_between(sm, glob_borders):
            good = True
    
    return res, good


comps = np.arange(len(samples))
    

results = []
for number in range(1,8): # how many different days by week
    for _ in range(10):
        inds = list(np.random.choice(comps, min(number, len(comps)), replace = False))
        smpl, flag = coef_sum(inds)
        if flag:
            print(smpl)
        
        results.append((inds, smpl))


unique_results = []
uniqs = []
for p in results:
    if p[0] not in uniqs:
        unique_results.append(Weeks([candidates[k] for k in p[0]],p[1]))
        uniqs.append(p[0])


for r in unique_results:
    for _, val in r.configurations:
        print(f'{score(val)}  {np.sum(val < borders[2,:])}   {np.sum(val > borders[3,:])}')
    print()
















