'''
information.py
'''

import numpy as _np
import pdb as _pdb

def getEntropy(prob):
    '''
    given a probability distribution (prob), calculate and return its entropy
    '''
    
    if not isinstance(prob, _np.ndarray):
        raise TypeError('getEntropy needs prob to be an ndarray') 

    errorVal = .001
    if abs(prob.sum()-1) > errorVal:
        raise ValueError('in getEntropy prob should sum to 1')
    
    # compute the log2 of the probability and change any -inf by 0s
    logProb = _np.log2(prob)
    logProb[logProb==-_np.inf] = 0
    
    # return dot product of logProb and prob
    return -1.0* _np.dot(prob, logProb)


def labels2prob(*argv):
    '''
    zipping all iterables in iterList, generate a dictionary with the probability of each item

    input:
        argv:   a list of iterables
                can be a single iterable
    '''
    from collections import Counter
    myCounter = Counter()
    
    N = len(argv[0])
    for item in zip(*argv):
        myCounter[item]+=1/N

    return myCounter


def mi(x, y):
    '''
    compute and return the mutual information between x and y
    
    inputs:
    -------
        x, y:   iterables with discrete symbols
    
    output:
    -------
        mi:     float
    '''
    #_pdb.set_trace()
    # dict.values() returns a view object that has to be converted to a list before being converted to an array
    probX = _np.array(list(labels2prob(x).values()))
    probY = _np.array(list(labels2prob(y).values()))
    probXY = _np.array(list(labels2prob(x, y).values()))

    return getEntropy(probX) + getEntropy(probY) - getEntropy(probXY)
