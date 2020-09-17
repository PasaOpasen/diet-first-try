# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:17:26 2020

@author: qtckp
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from weeksum import get7sum
from joblib import Parallel, delayed
import json

from loading import get_data


np.set_printoptions(precision=3, suppress = True)


# Little functions


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


def AE(sample, target):
    """
    сумма процентных отклонений от цели
    """
    return np.sum(np.abs(sample-target)/target)    



def get_dict(names, values):
    return {n: v for n, v in zip(names, values)}

# Classes


class Day:
    def __init__(self, recipes_weights, food_weights, combination = 0, less_than_down = None):
        self.recipes_weights = recipes_weights
        self.food_weights = food_weights
        self.combination = combination
        self.less_than_down = less_than_down
        
    def show_info(self):
        
        ri = {i: r for i, r in enumerate(self.recipes_weights) if r != 0}
        fi = {i: f for i, f in enumerate(self.food_weights) if f != 0}
        
        print('recipes index: count')
        print(ri)
        print()
        print('food index: count')
        print(fi)
        print()
        print(f'result: {self.combination} with {self.less_than_down} lesses under lower border')
    
    def to_dictionary(self, indexes):
        
        answer = {
            'recipes':[],
            'foods':[],
            'combination': get_dict(indexes['goal_columns'],self.combination), #self.combination.tolist(),
            'lower_error': int(self.less_than_down)
                  }
        
        for i, r in enumerate(self.recipes_weights):
            if r != 0:
                answer['recipes'].append({
                    'index': indexes['recipes_names'][i],
                    'count': int(r)
                    })
                
        for i, f in enumerate(self.food_weights):
            if f != 0:
                answer['foods'].append({
                    'index': indexes['foods_names'][i],
                    'count': int(f)
                    }) 
        
        return answer
    
    def to_json(self, file_name, indexes):
        
        dictionary = self.to_dictionary(indexes)
        
        with open(file_name, "w") as write_file:
            json.dump(dictionary, write_file, indent = 4)
            
    def plot(self, file_name, indexes, borders):
        
        df = pd.DataFrame({
            'nutrients': indexes['goal_columns'],
            'current result': self.combination/borders[1,:]*100,
            'lower border': borders[0,:]/borders[1,:]*100,
            'upper border': borders[1,:]/borders[1,:]*100
            })
        
        fig, ax = plt.subplots()
        

        
        
        df.plot(kind= 'line', x='nutrients', y='upper border', ax = ax, color = 'red', marker = 'o')
        
        df.plot(kind= 'line', x='nutrients', y='lower border', ax = ax, color = 'black', marker = 'o')
        
        df.plot(kind='bar',x='nutrients', y='current result', ax = ax)
        
        ax.set_xticklabels(df['nutrients'], rotation=90)
        
        plt.savefig(file_name, dpi = 350, bbox_inches = "tight")
        
        plt.close()
        
    
    def plot2(self, file_name, indexes, borders, foods, recipes):
        
        df = pd.DataFrame({
            'nutrients': indexes['goal_columns'],
            'current result': self.combination/borders[1,:]*100,
            'by recipes': self.recipes_weights.dot(recipes)/borders[1,:]*100,
            'by foods': self.food_weights.dot(foods)/borders[1,:]*100,
            'lower border': borders[0,:]/borders[1,:]*100,
            'upper border': borders[1,:]/borders[1,:]*100
            })
        
        
        print(np.allclose(df['current result'].values, (df['by recipes'] + df['by foods']).values))
        
        df['by foods'] = df['by recipes'] + df['by foods']
        
        fig, ax = plt.subplots()
        

        df.plot(kind= 'line', x='nutrients', y='upper border', ax = ax, color = 'red', marker = 'o')
        
        df.plot(kind= 'line', x='nutrients', y='lower border', ax = ax, color = 'black', marker = 'o')
        
        df.plot(kind='bar',x='nutrients', y='by foods', ax = ax)
        df.plot(kind='bar',x='nutrients', y='by recipes', ax = ax, color="C2")
        
        ax.set_xticklabels(df['nutrients'], rotation=90)
        
        plt.savefig(file_name, dpi = 350, bbox_inches = "tight")
        
        plt.close()

                
        

class Weeks:
    def __init__(self, days, configs, score, lower, upper):
        self.days = days
        self.configurations = configs
        self.score = score
        self.lower = lower
        self.upper = upper
        
    def show_info(self):
        print(f'I have {len(self.days)} days')
        print(f'My combination: {self.configurations[0]}')
        print(f'Vector: {self.configurations[1]}')
        print(f'score = {self.score}')
        print(f'lower error is {self.lower}, upper error is {self.upper}')
        
    def to_dictionary(self, indexes):
        answer = {
            
            'vector_of_combination': get_dict(indexes['goal_columns'], self.configurations[1]), #self.configurations[1].tolist(),
            'score': self.score,
            'lower_error': int(self.lower),
            'upper_error': int(self.upper),
            'days_in_week': [],
                  }
        
        for day , count in zip(self.days, self.configurations[0]):
            answer['days_in_week'].append({
                'day': day.to_dictionary(indexes),
                'repeats_in_week': int(count)
                })
        
        return answer
    
    def to_json(self, file_name, indexes):
        dictionary = self.to_dictionary(indexes)
        
        with open(file_name, "w") as write_file:
            json.dump(dictionary, write_file, indent = 4)
        



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


# как 3, но более хорошо ограничивает число разных рецептов (если указать 10,  часто по факту будет 10, 9 вместе 4-5 у третьего)

def get_day_fullrandom4(foods, recipes, borders, recipes_samples = 10, max_count = 3, tryes = 10):  
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




# как 4, но при подборе foods в первую очередь дает приоритет попыткам с меньшим количеством недотягиваний по колонкам

def get_day_fullrandom5(foods, recipes, borders, recipes_samples = 10, max_count = 3, tryes = 10): 
    
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
    print(f'{prob_food_inds.size} <= {foods.shape[0]}')
    
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
    #return [get_day_fullrandom6(foods, recipes, borders, recipes_samples, max_count, tryes) for _ in range(count)]
    return Parallel(n_jobs=6)(delayed(get_day_fullrandom6)(foods, recipes, borders, recipes_samples, max_count, tryes) for _ in range(count))

def get_optimal_candidates(foods, recipes, borders, recipes_samples = 4, max_count = 3, tryes = 10, count = 100, max_error_count = 3):
    cands = get_candidates(foods, recipes, borders, recipes_samples, max_count, tryes, count)
    return [cand for cand in cands if cand.less_than_down <= max_error_count]




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





def get_optimal_weeks(candidates, borders, lower_error = 3, upper_error = 3, limit = 7):

    
    samples = [res.combination for res in candidates]
    
    glob_borders = borders[2:4, :]
    
    avg = borders[4,:] 
    
    score = lambda sample: AE(sample, avg)
    
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
    
    
candidates = get_optimal_candidates(foods, recipes, borders, recipes_samples = 10, max_count = 3, tryes = 10, count = 100, max_error_count = 3)


for i, c in enumerate(candidates):
    c.to_json(f'results/day {i+1}.json', indexes)
    c.plot2(f'results/day {i+1}.png', indexes, borders, foods, recipes)


#weeks = get_optimal_weeks(candidates, borders, lower_error = 3, upper_error = 3, limit = 7)
    
    
# for i, week in enumerate(weeks):
#     week.show_info()
#     week.to_json(f'results/week {i+1}.json', indexes)
#     print()











