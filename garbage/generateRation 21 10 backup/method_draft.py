# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:17:26 2020

@author: qtckp
"""

import math
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import json
import warnings
from collections import namedtuple


import os, sys
sys.path.append(os.path.dirname(__file__))
# my modules
from weeksum import get7sum
from loading import get_data
from little_functions import currect_diff, is_between, is_valid_diff, will_be_valid_diff, MAPE
from classes import Day, Weeks
from coefficients import get_coefs_depended_on_goal, df_dot


np.set_printoptions(precision=3, suppress = True)


# как 7, но возращает первый же элемент с ошибкой не больше return_first_with_error

def get_day_fullrandom8(foods, recipes, borders, recipes_samples = 10, max_count = 3, tryes = 10, return_first_with_error = 4):  
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
            
                # new_bord = currect_diff(bord, recipes_used[i,:])
    
                # if is_valid_diff(new_bord):
                #     bord = new_bord
                #     counts[i] += 1
                #     good_results += 1
                # else:
                #     valid_flags[i] = False
                #     no_progress += 1
    
                if will_be_valid_diff(bord, recipes_used[i,:]):
                    bord = currect_diff(bord, recipes_used[i,:])
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
                    
                    # new_bord = currect_diff(bord, foods[food_inds[i],:])
                    
                    # if is_valid_diff(new_bord):
                    #     bord = new_bord
                    #     counts2[i] += 1
                    #     progress = True
                    # else:
                    #     break
                    
                    if will_be_valid_diff(bord, foods[food_inds[i],:]):
                        bord = currect_diff(bord, foods[food_inds[i],:])
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
                
                if errors <= return_first_with_error:
                    break
                
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




# как 8, но ещё есть ограничение на количество еды

def get_day_fullrandom9(foods, recipes, borders, recipes_samples = 10, max_count = 3, max_food_count = 15, tryes = 10, return_first_with_error = 4):  
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
            
                # new_bord = currect_diff(bord, recipes_used[i,:])
    
                # if is_valid_diff(new_bord):
                #     bord = new_bord
                #     counts[i] += 1
                #     good_results += 1
                # else:
                #     valid_flags[i] = False
                #     no_progress += 1
    
                if will_be_valid_diff(bord, recipes_used[i,:]):
                    bord = currect_diff(bord, recipes_used[i,:])
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
                    
                    # new_bord = currect_diff(bord, foods[food_inds[i],:])
                    
                    # if is_valid_diff(new_bord):
                    #     bord = new_bord
                    #     counts2[i] += 1
                    #     progress = True
                    # else:
                    #     break
                    
                    if will_be_valid_diff(bord, foods[food_inds[i],:]):
                        bord = currect_diff(bord, foods[food_inds[i],:])
                        counts2[i] += 1
                        progress = True
                        
                        if counts2[i] == max_food_count:
                            break
                        
                    else:
                        break
                    
            val = np.sum(bord[0,:]/borders[0, :])
            err = np.sum(bord[0,:] > 0)
            
            if err < errors:
                best_count2 = np.zeros(foods.shape[0])
                best_count2[food_inds] = counts2.copy()
                errors = err
                minval = val
                
                if errors <= return_first_with_error:
                    break
                
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
    return [get_day_fullrandom8(foods, recipes, borders, recipes_samples, max_count, tryes) for _ in range(count)]
    #return Parallel(n_jobs=6)(delayed(get_day_fullrandom7)(foods, recipes, borders, recipes_samples, max_count, tryes) for _ in range(count))

def get_optimal_candidates(foods, recipes, borders, recipes_samples = 4, max_count = 3, tryes = 10, count = 100, max_error_count = 3):
    cands = get_candidates(foods, recipes, borders, recipes_samples, max_count, tryes, count)
    return [cand for cand in cands if cand.less_than_down <= max_error_count]




def get_drinks_ways(needed_water, day, drinks, borders, indexes, max_drinks_samples = 4, max_count = 3, count = 10, max_iterations = 100):
    
    #current_border = currect_diff(borders, day.combination)[:2,:4] # only day borders and energy protein fat carb
    current_border = currect_diff(borders, day.combination)[:2,:] # only day borders and energy protein fat carb
    
    needed_water -= np.sum(indexes['water']['recipes'] * day.recipes_weights) + np.sum(indexes['water']['foods'] * day.food_weights)
    
    if needed_water < 0:
        return None
    
    
    max_sum = max_drinks_samples * max_count * 90 / 2
    if needed_water > max_sum:
        warnings.warn(f"WARNING.........{max_drinks_samples} drinks {max_count} times can get sum over {max_sum} < {needed_water} (needed additional sum). U can try to use more drinks or more count for drinks")
    
    
    drinks_names_list = indexes['drinks_names']
    drinks_names = np.array(drinks_names_list)
    
    
    inds = np.arange(drinks.shape[0])
    
    inds = inds[np.sum(drinks > current_border[1,:], axis = 1) == 0]
    
    max_drinks_samples = min(max_drinks_samples, inds.size)
    
    result = {}
    
    df_from_foods_and_recipes = df_dot(
        [indexes['start_dataframes']['recipes'],indexes['start_dataframes']['foods']],
        (day.recipes_weights, day.food_weights)
        )
    
    
    k = 0
    it = 0
    while True:
        
        total = 0
    
        drinks_inds = np.random.choice(inds, max_drinks_samples, replace = False)
        
        drinks_used = drinks[drinks_inds,:]
            
        counts = np.zeros(max_drinks_samples)
        
        bord = current_border.copy()
        
        for i in range(max_drinks_samples):
            for _ in range(max_count):
                
                if will_be_valid_diff(bord, drinks_used[i,:]):
                    bord = currect_diff(bord, drinks_used[i,:])
                    counts[i] += 1
                    total += indexes['water']['drinks'][drinks_inds[i]]
                    #print(total)
                    if total >= needed_water:
                        break
                else:
                    break
            
            if total >= needed_water:
                break
                 
            
        
        if True:#total >= needed_water:
            k += 1
            
            drks = {str(name): count for name, count in zip(drinks_names[drinks_inds], counts) if count > 0}
            
            coef_df = sum([indexes['start_dataframes']['drinks'].iloc[drinks_names_list.index(name),:].values * (count) for name,count in drks.items()])
            
            result[f'sample_{k}'] = {
                'drinks': drks,
                'additional_water': total,
                'coefficient': get_coefs_depended_on_goal(
                    #coef_df.to_frame().T + df_from_foods_and_recipes, 
                    pd.DataFrame(coef_df + df_from_foods_and_recipes.values, columns = df_from_foods_and_recipes.columns),
                    indexes['start_dataframes']['goal'])[0]
                }
            
        it += 1
        
        if k == count or it == max_iterations:
            break
    
    if len(result) == 0:
        result['total'] = {
            'additional_water': needed_water
            }
          
    
    return result, needed_water - total
    







if __name__ == '__main__':

    
    np.random.seed(5)
    
    
    
    foods, recipes, borders, drinks, indexes = get_data('norms.txt')
    
    
    
    # for _ in range(20):
    #     d = get_day_fullrandom3(foods, recipes, borders, 4, 3, 10)
    #     print(d.less_than_down)
        
    
        
    candidates = get_optimal_candidates(foods, recipes, borders, recipes_samples = 10, max_count = 3, tryes = 10, count = 100, max_error_count = 14)
    
    
    for i, c in enumerate(candidates):
        c.drinks = get_drinks_ways(3000, c, drinks, borders, indexes, max_drinks_samples = 4, max_count = 10, count = 10)
        c.to_json(f'results/day {i+1}.json', indexes)
        c.plot2(f'results/day {i+1}.png', indexes, borders, foods, recipes)
    
    
    
    
    
    
    
    #weeks = get_optimal_weeks(candidates, borders, lower_error = 4, upper_error = 4, valid_different_days = [1,2,3,4,5,6,7])
        
        
    # for i, week in enumerate(weeks):
    #     week.show_info()
    #     week.to_json(f'results/week {i+1}.json', indexes)
    #     print()
    
    
    
    
    






