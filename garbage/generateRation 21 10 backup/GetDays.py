"""
Created on Tue Sep 29 12:14:13 2020

@author: qtckp
"""

import sys, os
import math
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import json
import warnings
from collections import namedtuple

# my modules
sys.path.append(os.path.dirname(__file__))
from weeksum import get7sum
from loading import get_data
from little_functions import currect_diff, is_between, is_valid_diff, will_be_valid_diff, MAPE
from classes import Day, Weeks
from split_day import splitter_map

from method_draft import get_day_fullrandom9, get_drinks_ways


WATER_ID = 828

# это варианты суммы из условий

SUMS = {
    '3x': [25, 45, 30],
    '4x': [25, 35, [15, 25]],
    '5x': [[25, 10], [35, 10], 20],
    '6x': [[20, 10], [35, 10], [20, 5]],
    'night': [[10, 15], 25, [30, 15]]
}


def get_optimal_days(patient, prefix='',
                     how_many_days=7, max_tryes_for_days=150,
                     recipes_samples=12, max_recipes_count=3, max_food_count=15, food_tryes=10, 
                     return_first_with_error=3, max_error_count=13,
                     sums=[[15, 10], 40, 35],
                     max_drinks_samples=None, max_drinks_count=8, drinks_samples_count=10, max_iterations_for_drinks=100,
                     save_to_json=False, plot=False,
                     return_as_json=True):
    """
    Возращает оптимальные дни при нужных условиях

    Parameters
    ----------
    patient : строка, optional
        название файла с нормами или его содержание как json. The default is 'norms.txt'.
    prefix : строка, optional
        префикс, который будет добавлен в названия файлов. The default is ''.
    how_many_days : int, optional
        как много дней нужно сгенерировать. The default is 6.
    max_tryes_for_days : int, optional
        максимальное число попыток для получения первого дня; если прошло столько попыток, а день не найден, выдаётся исключение. The default is 100.
    recipes_samples : int, optional
        сколько разных рецептов допустимо использовать в одном дне. The default is 10.
    max_recipes_count : int, optional
        сколько максимально раз может повторяться каждый рецепт в одном дне. The default is 3.
    max_food_count : int, optional
        сколько максимально раз может повторяться каждый food в одном дне. The default is 15.
    food_tryes : int, optional
        столько раз для каждого дня ищется оптимальный расклад по foods, потом выбирается лучший. Чем меньше, тем быстрее, но хуже. The default is 10.
    return_first_with_error : int, optional
        как только возникает комбинация с таким числом несоответствий, она возвращается как оптимальная и поиски прекращаются. The default is 3.
    max_error_count : int, optional
        какое число несоотвествий по цели считается допустимым. The default is 4.
    sums : TYPE, optional
        суммы процентов для разбиения по приёмам пищи. The default is [[15,10], 40, 35].
    max_drinks_samples : int, optional
        сколько разных напитков допустимо использовать в одном дне. Если None, то столько же, сколько есть приёмов пищи. The default is None.
    max_drinks_count : int, optional
        сколько раз можно использовать каждый напиток в одном дне. The default is 5.
    drinks_samples_count : int, optional
        сколько конфигураций по напиткам нужно вернуть (сколько разных вариантов). The default is 10.
    max_iterations_for_drinks : int, optional
        число попыток для поиска указанного числа конфигураций. The default is 100.
    save_to_json : bool, optional
        сохранять ли полученные дни в json (если True, каждый день в отдельности будет сохранён). The default is True.
    plot : bool, optional
        сохранять ли изображения. The default is True.
    return_as_json : bool, optional
        возвращать ли полученный словарь как строку json (иначе словарь python). The default is False.

    Raises
    ------
    Exception
        выдает сообщение, если не получилось найти хотя бы 1 день.

    Returns
    -------
    словарь как json или python (зависит от return_as_json), полученные дни как список экземпляров класса Day

    """

    # getting data
    foods, recipes, borders, drinks, indexes = get_data(patient)
    needed_water = indexes['water']['needed']

    # getting valid days
    days = []
    counter = 0

    while len(days) < how_many_days:

        cand = get_day_fullrandom9(foods, recipes, borders, recipes_samples,
                                   max_recipes_count, max_food_count, food_tryes, return_first_with_error)

        if cand.less_than_down <= max_error_count:
            days.append(cand)
        else:
            counter += 1

        if len(days) == 0 and counter == max_tryes_for_days:
            raise Exception(
                f'Не получилось найти хотя бы 1 валидный день за {max_tryes_for_days} попыток. Попробуйте смягчить условия')

    # getting drinks
            
    max_drinks_samples = max_drinks_samples if max_drinks_samples != None else sum((len(t) if type(t) == list else 1 for t in sums))
    for day in days:
        tmp = get_drinks_ways(needed_water, day, drinks, borders, indexes, max_drinks_samples,
                                     max_drinks_count, drinks_samples_count, max_iterations_for_drinks)
        
        if tmp == None:
            day.drinks = None
            continue
        # здесь напитки кидаются на приёмы пищи
        drinks_dictionary, water = tmp
        
        coefs = np.array([value['coefficient'] for _, value in drinks_dictionary.items()])
        
        best_drinks = drinks_dictionary[list(drinks_dictionary.keys())[np.argmax(coefs)]]['drinks']
        #print(best_drinks)
        inds = [1] if type(sums[0]) != list else [1, 2]
        inds += [3] if type(sums[1]) != list else [3, 4]
        inds += [5] if type(sums[2]) != list else [5, 6]
        
        empty_drinks = {splitter_map[str(i)]: {} for i in inds}
        
        for i, (key, val) in enumerate(best_drinks.items()):
            empty_drinks[splitter_map[str(inds[i])]][key] = val
        best_drinks = empty_drinks
        #print(best_drinks)
        
        if water > 0:
            
            # надо на завтрак, первый перекус и обед по возможности
            ranges = [
                range(3),
                range(2)
                ]
            if type(sums[0]) != list:
                ranges = [
                    [0, 2, 4],
                    [0, 2]
                    ]
            
            if water > 600:
                for nb in ranges[0]:
                    best_drinks[splitter_map[str(nb+1)]][str(WATER_ID)] = water/3/100 # деление на 100, чтобы в общие единицы перевести
            elif water > 400:
                for nb in ranges[1]:
                    best_drinks[splitter_map[str(nb+1)]][str(WATER_ID)] = water/2/100 # деление на 100, чтобы в общие единицы перевести
            else:
                best_drinks[splitter_map[str(1)]][str(WATER_ID)] = water/100 # деление на 100, чтобы в общие единицы перевести
            
        
        day.drinks = best_drinks
        
        # print(sums)
        # print(best_drinks)
        # print()
        
        
        
    folder_name = 'results'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  
    
    
    # getting split and save to json
    if save_to_json:
             
        for i, day in enumerate(days):

            day.to_json(f'{folder_name}/{prefix} day {i+1}.json', indexes, sums)

    if plot:
        for i, day in enumerate(days):
            day.plot2(f'{folder_name}/{prefix} day {i+1}.png',
                      indexes, borders, foods, recipes)

    # return [day.to_dictionary(indexes, sums) for day in days]

    result_as_dictionary = {f'{prefix}_day_{i+1}': day.to_dictionary(
        indexes, sums, rewrite=False) for i, day in enumerate(days)}
    return json.dumps(result_as_dictionary) if return_as_json else result_as_dictionary, days, borders, indexes





WeekPair = namedtuple('WeekPair', 'combination amount_vector')
WeekCombination = namedtuple('WeekCombination', 'indexes weekpair')


def get_optimal_weeks(candidates, borders, indexes,
                      lower_error = 8, upper_error = 8, 
                      valid_different_days = [1,2,3,4,5,6,7], max_day_repeats = 2,
                      return_as_json = False):
    """
    возвращает словарь с валидными неделями

    Parameters
    ----------
    candidates : TYPE
        список из годный дней (второй элемент в кортеже от функции, что возвращает дни).
    borders : TYPE
        границы (третий аргумент той функции).
    indexes : TYPE
        четвёртый аргумент той функции.
    lower_error : TYPE, optional
        по скольким нутриентам можно не добрать до нижней границы. The default is 4.
    upper_error : TYPE, optional
        по скольким нутриентам можно превышать верхнюю границу. The default is 4.
    valid_different_days : TYPE, optional
        список чисел: сколько разных дней может встречаться за 1 неделю. The default is [1,2,3,4,5,6,7].
    max_day_repeats : TYPE, optional
        сколько максимум раз в неделе может повторяться 1 день. The default is 7.
    return_as_json : TYPE, optional
        вернуть как строку json (иначе возвращает словарь). The default is False.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    different_days = [count for count in valid_different_days if count <= 7 and count <= len(candidates)]
    
    if len(different_days) == 0:
        raise Exception(f'there are only {len(candidates)} different days what is less than each valid count in valid_different_days ({valid_different_days})')
    
    if len(different_days) < len(valid_different_days):
        warnings.warn(f"WARNING.........from valid_different_days ({valid_different_days}) it uses only {different_days} because there are only {len(candidates)} different days")
    
    
    samples = [res.combination for res in candidates]
    
    glob_borders = borders[2:4, :]
    
    avg = borders[4,:] 
    
    score = lambda sample: MAPE(sample, avg)
    
    weeks = get7sum(7, max_day_repeats)
    
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
            res.append(WeekPair(combination = arr, amount_vector = sm))
            if is_between(sm, glob_borders):
                good = True
        
        return res, good
    
    
    
    len_samples = len(samples)
    
    comps = np.arange(len_samples)
        
    
    # это кусок кода ищет для индексов от samples типа (1, 2, 3) разные комбинации этих рецептов по неделе типа [2, 4, 1], то есть два дня первый рецепт, четыре дня второй и один день третий
    
    choises_count = range(min(10, 2*math.factorial(len_samples)))
    
    results = []
    for number in different_days: # how many different days by week
        for _ in choises_count:
            inds = list(np.random.choice(comps, number, replace = False)) # столько-то индексов для массива samples
            smpl, flag = coef_sum(inds)
            if flag:
                print(smpl)
            
            results.append(WeekCombination(indexes = inds, weekpair = smpl))
    
    # убираются дубликаты и генерируются ответы
    
    unique_results = []
    uniqs = []
    for p in results:
        if p.indexes not in uniqs:
            for pair in p.weekpair:
            
                score_ = score(pair.amount_vector)
                lower = np.sum(pair.amount_vector < borders[2,:])
                upper = np.sum(pair.amount_vector > borders[3,:])
                
                if lower <= lower_error and upper <= upper_error:
                    unique_results.append(Weeks([candidates[k] for k in p.indexes], pair, score_, lower, upper))
                    uniqs.append(p.indexes)
    
    
    
    result_as_dictionary = {f'week_{i}': week.to_dictionary(indexes) for i, week in enumerate(unique_results)}
    
    return json.dumps(result_as_dictionary) if return_as_json else result_as_dictionary







if __name__ == '__main__':

    WANNA_SEE_DAYS = True
    
    if WANNA_SEE_DAYS:
    
        answer, days, borders, indexes = get_optimal_days(
            'norms.txt',
            prefix='default', save_to_json=False, plot = True, return_as_json=True)
    
        answers = {}
        for time, val in SUMS.items():
    
            answers[time], days, border, indexes = get_optimal_days('norms.txt', sums=val, prefix=time, save_to_json=True, plot = True)
        
    # weeks
        
        # getting weeks
        # 1) getting days
    answer, days, borders, indexes = get_optimal_days(
        'norms.txt',
        prefix='default', save_to_json=False, return_as_json=True)
        # 2) weeks from days
        
    weeks = get_optimal_weeks(days, borders, indexes, lower_error = 8, upper_error = 8, valid_different_days = [1,2,3,4,5,6,7], max_day_repeats = 7, return_as_json = False)
        
    with open('weeks.json', 'w') as file:
        json.dump(weeks, file, indent = 4)
    
    
    