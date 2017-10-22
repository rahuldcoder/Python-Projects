# -*- coding: utf-8 -*-
"""
@author: Rahul Thakur
"""
import numpy as np
import warnings
from collections import Counter

dataset ={'x':[[1,2],[2,3],[3,1]],'y':[[6,5],[7,7],[8,6]] }
new_features=[5,7]

def K_nearest_neighbours(data,predict,k=3):
    if len(data) >=k:
        warnings.warn('K is set to a value less than total voting groups')
    distances =[]
    for group in data:
        for features in data[group]:
            euclidean_distance =np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
            
    votes=[i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    votes_result =Counter(votes).most_common(1)[0][0]

    return votes_result
result =K_nearest_neighbours(dataset,new_features,k=3)
print(result)          