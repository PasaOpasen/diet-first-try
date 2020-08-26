# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:17:26 2020

@author: qtckp
"""

import numpy as np

np.set_printoptions(precision=3, suppress = True)


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

def is_valid_subsample(subsample, borders, maxcount):
    """
    если для какого-либо признака нет рецепта с таким условием, по логике алгоритма никакая комбинация таких рецептов не подойдёт
    
    к сожалению, не сработает
    """
    max_by_prop = np.max(subsample, axis = 0)
    #print(np.sum(max_by_prop >= borders[0,:]/(subsample.shape[0]*maxcount)))
    return np.sum(max_by_prop >= borders[0,:]/(subsample.shape[0]*maxcount)) == borders.shape[1]

def is_between(sample, borders):
    
    if np.sum(sample < borders[0,:]) != 0:
        return False
    
    if np.sum(sample > borders[1,:]) != 0:
        return False
    
    return True


def AE(sample, target):
    return np.sum(np.abs(sample-target))    


def get_day(foods, recipes, borders, recipes_samples = 10, max_count = 3):
    
    # recipes part
    
    recipes_inds = np.random.choice(recipes.shape[0], recipes_samples)
    
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
    
    # foods part
    
    food_size = foods.shape[0]
    
    food_inds = np.arange(food_size)
    np.random.shuffle(food_inds)


    counts2 = np.zeros(food_size)
    
    for i in range(food_size):
        
        while True:
            new_bord = currect_diff(bord, foods[food_inds[i],:])
            
            if is_valid_diff(new_bord):
                bord = new_bord
                counts2[i] += 1
            else:
                break
    
    
    
    # currect weights
            
    recipes_weights = np.zeros(recipes.shape[0])
    recipes_weights[recipes_inds] = counts
    #print(recipes_weights)
    
    food_weights = np.zeros(food_size)
    food_weights[food_inds] = counts2
    #print(food_weights)
    
    # results
    
    r = np.sum(recipes * recipes_weights.reshape(recipes.shape[0], 1), axis = 0)
    f = np.sum(foods * food_weights.reshape(food_size, 1), axis = 0)
    
    
    #print((r + f) / borders[0,:])
    #print(r + f < borders[0,:])
    #print(r + f)
    #print(borders[0,:])
    
    
    
    #return r+f
    
    return np.sum(r + f < borders[0,:])
    
    assert(np.sum(r + f < borders[0,:]) == 0)
    
    # это условие всегда выполнено из смысла самого алгоритма
    assert(np.sum(r + f > borders[1,:]) == 0)

def get_day2(foods, recipes, borders, recipes_samples = 10, max_count = 3, tryes = 10):
    
    # recipes part
    
    recipes_inds = np.random.choice(recipes.shape[0], recipes_samples)
    
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
        
        for i in range(food_size):
            
            while True:
                new_bord = currect_diff(bord, foods[food_inds[i],:])
                
                if is_valid_diff(new_bord):
                    bord = new_bord
                    counts2[i] += 1
                else:
                    break
                
        val = np.sum(bord[0,:])
        if val < minval:
            best_count2 = counts2.copy()
            minval = val
        
    
    counts2 = best_count2
    
    
    # currect weights
            
    recipes_weights = np.zeros(recipes.shape[0])
    recipes_weights[recipes_inds] = counts
    #print(recipes_weights)
    
    food_weights = np.zeros(food_size)
    food_weights[food_inds] = counts2
    #print(food_weights)
    
    # results
    
    r = np.sum(recipes * recipes_weights.reshape(recipes.shape[0], 1), axis = 0)
    f = np.sum(foods * food_weights.reshape(food_size, 1), axis = 0)
    
    
    #print((r + f) / borders[0,:])
    #print(r + f < borders[0,:])
    #print(r + f)
    #print(borders[0,:])
    
    
    
    #return r+f
    
    return np.sum(r + f < borders[0,:])
    
    assert(np.sum(r + f < borders[0,:]) == 0)
    
    # это условие всегда выполнено из смысла самого алгоритма
    assert(np.sum(r + f > borders[1,:]) == 0)

def get_day3(foods, recipes, borders, recipes_samples = 10, max_count = 3, tryes = 10):
    
    # recipes part
    while True:
        recipes_inds = np.random.choice(recipes.shape[0], recipes_samples)
        
        recipes_used = recipes[recipes_inds,:]
        
        if is_valid_subsample(recipes_used, borders, max_count):
            break
        
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
        
        for i in range(food_size):
            
            while True:
                new_bord = currect_diff(bord, foods[food_inds[i],:])
                
                if is_valid_diff(new_bord):
                    bord = new_bord
                    counts2[i] += 1
                else:
                    break
                
        val = np.sum(bord[0,:])
        if val < minval:
            best_count2 = counts2.copy()
            minval = val
        
    
    counts2 = best_count2
    
    
    # currect weights
            
    recipes_weights = np.zeros(recipes.shape[0])
    recipes_weights[recipes_inds] = counts
    #print(recipes_weights)
    
    food_weights = np.zeros(food_size)
    food_weights[food_inds] = counts2
    #print(food_weights)
    
    # results
    
    r = np.sum(recipes * recipes_weights.reshape(recipes.shape[0], 1), axis = 0)
    f = np.sum(foods * food_weights.reshape(food_size, 1), axis = 0)
    
    
    #print((r + f) / borders[0,:])
    #print(r + f < borders[0,:])
    #print(r + f)
    #print(borders[0,:])
    
    
    
    #return r+f
    
    return np.sum(r + f < borders[0,:])
    
    assert(np.sum(r + f < borders[0,:]) == 0)
    
    # это условие всегда выполнено из смысла самого алгоритма
    assert(np.sum(r + f > borders[1,:]) == 0)



def get_day_fullrandom(foods, recipes, borders, recipes_samples = 10, max_count = 3, tryes = 10):
    
    # recipes part
    
    recipes_inds = np.random.choice(recipes.shape[0], recipes_samples)
    
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
        
        for i in range(food_size):
            
            while True:
                new_bord = currect_diff(bord, foods[food_inds[i],:])
                
                if is_valid_diff(new_bord):
                    bord = new_bord
                    counts2[i] += 1
                else:
                    break
                
        val = np.sum(bord[0,:])
        if val < minval:
            best_count2 = counts2.copy()
            minval = val
        
    
    counts2 = best_count2
    
    
    # currect weights
            
    recipes_weights = np.zeros(recipes.shape[0])
    recipes_weights[recipes_inds] = counts
    #print(recipes_weights)
    
    food_weights = np.zeros(food_size)
    food_weights[food_inds] = counts2
    #print(food_weights)
    
    # results
    
    r = np.sum(recipes * recipes_weights.reshape(recipes.shape[0], 1), axis = 0)
    f = np.sum(foods * food_weights.reshape(food_size, 1), axis = 0)
    
    score = r + f
    
    return score, np.sum(score < borders[0,:]), recipes_weights, food_weights
    


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




candidates = []
for _ in range(50):
    cand = get_day_fullrandom(foods, recipes, borders[0:2,:], 4, 3, 10)
    if cand[1] < 4:
        candidates.append(cand)


for i in range(len(candidates)-1):
    for j in range(i+1, len(candidates)):
        score = (candidates[i][0] + candidates[j][0])/2
        print(np.sum(score < borders[0,:]))


# for p in range(500):
#     bl = get_day3(foods, recipes, borders[0:2,:], 4, 3, 10)
#     if bl < 5:
#         print(f'{p}   {bl}')



# for p in range(500):
#     a = get_day2(foods, recipes, borders[0:2,:], 4, 3, 10)
#     b = get_day2(foods, recipes, borders[0:2,:], 4, 3, 10)
#     #c = get_day2(foods, recipes, borders[0:2,:], 4, 3, 10)
#     if np.sum((a+b)/2 < borders[0,:]) <= 2:
#         print(f'{p}   {np.sum((a+b)/2 > borders[1,:])}')



# pred_count = 80

# food_wrap = np.random.uniform(low = 0.5, high = 3, size = (100, pred_count))
# recipes_wrap = np.random.uniform(low = 2, high = 4, size = (150, pred_count))


# a = np.random.normal(loc = 50, scale = 5, size = pred_count)
# b = np.random.normal(loc = 60, scale = 3, size = pred_count)

# borders_wrap  = np.vstack((
#         a,
#         a + np.random.uniform(low = 5, high = 10, size = pred_count),
#         b,
#         b + np.random.uniform(low = 1.5, high = 3, size = pred_count)
#     ))

# np.sum(borders_wrap [3,:] > borders_wrap [2,:])
# np.sum(borders_wrap [1,:] > borders_wrap [0,:])



# get_day(food_wrap , recipes_wrap , borders_wrap[0:2,:], 7)




# import sympy
# mat = np.array([[0,1,0,0],[0,0,1,0],[0,1,1,0],[1,0,0,1]]) 
# _, inds = sympy.Matrix(mat).T.rref() 


# mat = recipes
# _, inds = sympy.Matrix(mat).T.rref() 


































