# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:45:14 2020

@author: qtckp
"""

import math
import numpy as np
from collections import defaultdict

def get_sums_by_classes(vals, classes, classes_count = None):
    
    if classes_count == None:
        tot = np.unique(classes)
        classes_count = tot.size
    else: 
        tot = np.arange(classes_count)
        
    res = defaultdict(float)
    
    for v, c in zip(vals, classes):
        res[c] += v
    
    return {k: v for k, v in sorted(res.items(), key=lambda item: item[0])}

def convert_sums_by_classes(dic, sums, tol = 10):
    res = defaultdict(float)
    flag = True
    
    for (k,v), s in zip(dic.items(), sums):
        
        res[k] = (v - s)/s*100
        
        if math.fabs(res[k])>tol:
           flag = False 
    
    return res, flag



def get_current_dic(vals, classes, sums, tol = 10, classes_count = None):
    
    return convert_sums_by_classes(get_sums_by_classes(vals, classes, classes_count), sums, tol)



def which_max(vals, classes, class_index):
    
    mx = float('-inf')
    
    for i, (v, c) in enumerate(zip(vals, classes)):
        if c == class_index:
            if v > mx:
                mx = v
                result = (v, i)
    return result






def get_split_by_sums(vals, prefer_classes, sums, tol = 10):
    
    sums = np.array(sums) * vals.sum()
    
    result_classes = prefer_classes.copy()
    
    dic, flag = get_current_dic(vals, result_classes, sums, tol, sums.size)
        
    if flag:
        return result_classes, list(dic.values())   
    
    procents = np.array(list(dic.values()))
    
    while not flag:
        
        print(procents)
        
        max_class = np.argmax(procents)
        min_class = np.argmin(procents)
        print(f'{max_class} {min_class}')
        value, index = which_max(vals, result_classes, max_class)
        
        #procents[max_class] -= value/sums[max_class]*100
        #procents[min_class] += value/sums[min_class]*100

        result_classes[index] = min_class
        
        dic, flag = get_current_dic(vals, result_classes, sums, tol, sums.size)
        procents = np.array(list(dic.values()))
        
        #flag = np.sum(np.abs(procents) > tol) == 0
    
    return result_classes, list(procents)






d = get_sums_by_classes(np.array([0.1,2,3,4,5]), np.array([1,2,3,2,1]))


convert_sums_by_classes(d, [3.0,7.0,1.0])



vals = np.random.uniform(low = 0.01, high = 100, size = 50)
classes = np.random.choice([0,1,2], 50, True)

ans, p = get_split_by_sums(vals, classes, [0.25, 0.25, 0.5], tol = 10)



get_current_dic(vals, ans, np.array([0.25, 0.25, 0.5]) * vals.sum(), tol = 10)





