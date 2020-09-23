# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:17:26 2020

@author: qtckp
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import json



# my modules
from weeksum import get7sum
from loading import get_data
from little_functions import currect_diff, is_between, is_valid_diff, will_be_valid_diff, MAPE
from classes import Day, Weeks



np.set_printoptions(precision=3, suppress = True)




# как 5, но при начальном наборе foods старается брать только с высоким содержанием нутриентов, которых не хватает до нижней границы

def get_day_fullrandom6(foods, recipes, borders, recipes_samples = 10, max_count = 3, tryes = 10):  
    rc = recipes_samples
    
    recipes_samples = rc*10
    
    # recipes part
    
    recipes_inds = np.random.choice(recipes.shape[0], recipes_samples, replace = False)
    
    recipes_used = recipes[recipes_inds,:]
        
    counts = np.zeros(recipes_samples)
    
    bord = borders[0:2,:].copy()

    valid_flags = np.ones(recipes_samples, dtype = np.bool)
    for _ in range(max_count):
        
        no_progress = 0
        good_results = 0
        
        for i in range(recipes_samples):
            
            if valid_flags[i]:
            
                new_bord = currect_diff(bord, recipes_used[i,:])
    
                if is_valid_diff(new_bord):
                    bord = new_bord
                    counts[i] += 1
                    good_results += 1
                else:
                    valid_flags[i] = False
                    no_progress += 1
            else:
                no_progress += 1
            
            if good_results == rc:
                for j in range(i+1, recipes_samples):
                    valid_flags[j] = False
                break
        
        if no_progress == recipes_samples:
            break
    
    #r = np.sum(recipes_used * counts.reshape(recipes_used.shape[0], 1), axis = 0)
    #print(np.sum(r > borders[1,:]) == 0)
    
    
    # foods part
    
    err_inds = [i for i, b in enumerate(bord[0,:]) if b > 0]
    
    low_foods = bord[0, err_inds]/10
    
    prob_food_inds = np.array([i for i, food in enumerate(foods) if np.sum(food>bord[1,:]) == 0 and np.sum(food[err_inds] >= low_foods)])
    #print(prob_food_inds)
    
    #print(f'{prob_food_inds.size} <= {foods.shape[0]}')
    
    if prob_food_inds.size == 0:
        food_weights = np.zeros(foods.shape[0])
        f = np.zeros(foods.shape[1])
    else:
    
        food_size = prob_food_inds.size
        
        food_inds = prob_food_inds.copy() #np.arange(food_size)
        
        minval = float('inf')
        errors = foods.shape[1]
        
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
            err = np.sum(bord[0,:] > 0)
            
            if err < errors:
                best_count2 = np.zeros(foods.shape[0])
                best_count2[food_inds] = counts2.copy()
                errors = err
                minval = val
            elif err == errors and val < minval:
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
    return [get_day_fullrandom6(foods, recipes, borders, recipes_samples, max_count, tryes) for _ in range(count)]
    #return Parallel(n_jobs=6)(delayed(get_day_fullrandom6)(foods, recipes, borders, recipes_samples, max_count, tryes) for _ in range(count))

def get_optimal_candidates(foods, recipes, borders, recipes_samples = 4, max_count = 3, tryes = 10, count = 100, max_error_count = 3):
    cands = get_candidates(foods, recipes, borders, recipes_samples, max_count, tryes, count)
    return [cand for cand in cands if cand.less_than_down <= max_error_count]





def get_optimal_weeks(candidates, borders, lower_error = 3, upper_error = 3, limit = 7):

    
    samples = [res.combination for res in candidates]
    
    glob_borders = borders[2:4, :]
    
    avg = borders[4,:] 
    
    score = lambda sample: MAPE(sample, avg)
    
    weeks = get7sum(limit)
    
    up_lim = max(weeks.keys())
    
    def coef_sum(inds):
        """
        принимает на вход индексы образцов из samples
        
        считает комбинации этих samples по weeks
        
        возвращает [(комбинация дней из weeks, вектор суммы)], (есть ли суперудачный ответ хотя бы в одной комбинации)
        """
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
    
    
    len_samples = len(samples)
    
    
    comps = np.arange(len_samples)
        
    
    # это кусок кода ищет для индексов от samples типа (1, 2, 3) разные комбинации этих рецептов по неделе типа [2, 4, 1], то есть два дня первый рецепт, и дня второй и один день третий
    
    choises_count = range(min(10, math.factorial(len_samples)))
    
    results = []
    for number in range(1,8): # how many different days by week
        for _ in choises_count:
            inds = list(np.random.choice(comps, min(number, len_samples), replace = False)) # столько-то индексов для массива samples
            smpl, flag = coef_sum(inds)
            if flag:
                print(smpl)
            
            results.append((inds, smpl))
    
    # убираются дубликаты и генерируются ответы
    
    unique_results = []
    uniqs = []
    for p in results:
        if p[0] not in uniqs:
            for pair in p[1]:
            
                score_ = score(pair[1])
                lower = np.sum(pair[1] < borders[2,:])
                upper = np.sum(pair[1] > borders[3,:])
                
                if lower <= lower_error and upper <= upper_error:
                    unique_results.append(Weeks([candidates[k] for k in p[0]], pair, score_, lower, upper))
                    uniqs.append(p[0])
    
    return unique_results









np.random.seed(5)



foods, recipes, borders, indexes = get_data()



# for _ in range(20):
#     d = get_day_fullrandom3(foods, recipes, borders, 4, 3, 10)
#     print(d.less_than_down)
    

    
candidates = get_optimal_candidates(foods, recipes, borders, recipes_samples = 10, max_count = 3, tryes = 10, count = 100, max_error_count = 4)


for i, c in enumerate(candidates):
    c.to_json(f'results/day {i+1}.json', indexes)
    c.plot2(f'results/day {i+1}.png', indexes, borders, foods, recipes)


#weeks = get_optimal_weeks(candidates, borders, lower_error = 3, upper_error = 3, limit = 7)
    
    
# for i, week in enumerate(weeks):
#     week.show_info()
#     week.to_json(f'results/week {i+1}.json', indexes)
#     print()











