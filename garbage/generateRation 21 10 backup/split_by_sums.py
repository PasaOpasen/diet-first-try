# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:45:14 2020

@author: qtckp
"""

import math
import numpy as np
from collections import defaultdict

def get_sums_by_classes(vals, classes):
    """
    возвращает суммы по каждому классу в виде словаря
    """
        
    res = defaultdict(float)
    
    for v, c in zip(vals, classes):
        res[c] += v
    
    return {k: v for k, v in sorted(res.items(), key=lambda item: item[0])}

def convert_sums_by_classes(dic, sums, tol = 10):
    """
    преобразует словарь сумм в словарь процентных отклонений от цели
    """
    res = defaultdict(float)
    flag = True
    
    for (k,v), s in zip(dic.items(), sums):
        
        res[k] = (v - s)/s*100
        
        if math.fabs(res[k])>tol:
           flag = False 
    
    return res, flag



def get_current_dic(vals, classes, sums, tol = 10):
    """
    Считает суммы по каждому классу и преобразует их в вид процентных отклонений от цели

    Parameters
    ----------
    vals : numpy array
        значения величин.
    classes : numpy array
        метки классов, соответствующие предыдущему аргументу.
    sums : numpy array
        массив целей разбиения по классам (в абсолютных величинах, типа [100, 80, 93]).
    tol : float, optional
        на какой процент в любой категории можно отклоняться. The default is 10.

    Returns
    -------
    dictionary, bool
        словарь процентных отклонений по классам и флаг, подходит ли разбиение уловию.

    """
    
    return convert_sums_by_classes(get_sums_by_classes(vals, classes), sums, tol)



def which_max(vals, classes, class_index):
    """
    возвращает максимальный элемент, принадлежащий указанному классу, и его индекс в общем массиве
    """
    mx = float('-inf')
    
    result = (mx, -1)
    #print(classes)
    #print(class_index)
    
    for i, (v, c) in enumerate(zip(vals, classes)):
        if c == class_index:
            if v > mx:
                mx = v #; print(v)
                result = (v, i)
    return result






def get_split_by_sums(vals, prefer_classes, sums, tol = 10, max_iter = 30):
    """
    ищет разбиение на классы, дающее сумму в классах в нужных диапазонах

    Parameters
    ----------
    vals : TYPE
        значения величин.
    prefer_classes : TYPE
        начальное разбиение на классы.
    sums : TYPE
        идеальные отношения сумм.
    tol : TYPE, optional
        на сколько процентов можно отклоняться. The default is 10.
    
    max_iter: максимальное число итераций, ибо может зацикливаться
    
    Returns
    -------
    TYPE
        подходящее разбиение и проценты отклонений по классам.

    """
    
    sums = np.array(sums) * vals.sum()
    
    result_classes = prefer_classes.copy()
    labels = np.unique(result_classes)
    
    dic, flag = get_current_dic(vals, result_classes, sums, tol)
    #print(dic)    
    if flag:
        return result_classes, list(dic.values())   
    
    procents = np.array(list(dic.values()))
    k = 0
    
    while not flag:
        
        #print(procents)
        
        max_class = labels[np.argmax(procents)]
        min_class = labels[np.argmin(procents)]
        #print(f'{max_class} {min_class}')
        value, index = which_max(vals, result_classes, max_class)
        
        #procents[max_class] -= value/sums[max_class]*100
        #procents[min_class] += value/sums[min_class]*100

        result_classes[index] = min_class
        
        dic, flag = get_current_dic(vals, result_classes, sums, tol)
        procents = np.array(list(dic.values()))
        k += 1
        
        # показывать промежуточные результаты
        # print(dic)
        
        if k == max_iter:
            return result_classes, []
        #flag = np.sum(np.abs(procents) > tol) == 0
    
    return result_classes, list(procents)





if __name__ == '__main__':

    # очень маленький пример
    d = get_sums_by_classes(np.array([0.1,2,3,4,5]), np.array([1,2,3,2,1]))
    
    print(d) 
    
    print(convert_sums_by_classes(d, [3.0,7.0,1.0]))
    
    
    
    vals = np.random.uniform(low = 0.01, high = 100, size = 50)
    classes = np.random.choice([0,1,2], 50, True)
    
    ans, p = get_split_by_sums(vals, classes, [0.25, 0.25, 0.5], tol = 10)
    
    print(ans)
    print(p)
    
    print(get_current_dic(vals, ans, np.array([0.25, 0.25, 0.5]) * vals.sum(), tol = 10))
    
    
    
    size = 100
    classes_count = 4
    
    vals = np.random.uniform(low = 0.01, high = 100, size = size)
    classes = np.random.choice(np.arange(classes_count), size, True)
    
    ans, p = get_split_by_sums(vals, classes, [0.25, 0.25, 0.2, 0.3], tol = 10)
    
    print(ans)
    print(p)





