# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:18:21 2020

@author: qtckp
"""


def get7sum(limit = 3):
    
    if limit > 7:
        limit = 7

    comps = list(range(1,7))
    
    results = []#[([1],1)]
    
    for c in comps:
        for _ in range(limit):
            tmp = [([c], c)]
            for r, s in results:
                if s + c <= 7:
                    tmp.append((r + [c], s + c))
            results += tmp
    
    tmp = [sorted(r) for r, s in results if s == 7]
    
    answer = []
    for t in tmp:
        if t not in answer:
            answer.append(t)
        
    #return answer
            
    dic = {}
    for i in range(1,8):
        dic[i] = []
    for a in answer:
        dic[len(a)].append(a)
    
    return dic

if __name__ == '__main__':
    get7sum(5)


