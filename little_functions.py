# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:35:25 2020

@author: qtckp
"""

import numpy as np




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


def MAPE(sample, target):
    """
    сумма процентных отклонений от цели
    """
    return np.mean(np.abs(sample-target)/target)    




