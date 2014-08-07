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

def gaussianEntropy(var):
    '''
    compute the entropy of a gaussian distribution with variance 'var' using differential entropy formula. Works for covariance matrix as well

    var:    float, the variance
            ndarray, a covariance matrix
    '''
    
    #print(var.ndim)
    #print(_np.linalg.det(var))
    #print(0.5*_np.log2( (2*_np.pi*_np.e)**var.ndim * _np.linalg.det(var) ))
    try:
        # assuming var is a covariance matrix (a ndarray of at least 2 dimensions)
        return 0.5*_np.log2( (2*_np.pi*_np.e)**var.ndim * _np.linalg.det(var) )
        # replace inf by 0
        #entropy[_np.where(entropy == inf)] = 0
        #return entropy
    except:
        # if it failed, then most likely is just a number
        # if var == 0, log2 will be inf but H should be 0
        #if var==0:
        #    return 0
        #else:
        return 0.5*_np.log2(2*_np.pi*_np.e * var)
