'''
information.py
'''

import numpy as _np
import pdb as _pdb

def getEntropy(prob):
    '''
    given a probability distribution (prob), calculate and return its entropy
    '''
    
    #_pdb.set_trace()

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


def labels_to_prob(labels):
    '''
    Return the probability distribution of labels. Only probabilities are returned and in random order, you don't know what the probability of a given label is but this can be used to compute entropy

    input:
        labels:     iterable of hashable items
                    works well if labels is a zip of iterables
    '''
    from collections import Counter
    myCounter = Counter
    
    #_pdb.set_trace()

    #if isinstance(labels, zip):
    #    labels = list(labels)

    #for i in zip(*argv):
    #    print(i, type(i))
    #    if ismutable(arg[0]):
    #        argv[i] = list(map(tuple, arg))

    
    # count number of occurrances of each simbol in *argv (return as list of just the count)
    asList = list(myCounter(labels).values())

    # total count of symbols
    N = sum(asList)

    return _np.array([n/N for n in asList])

def combine_labels(*args):
    return tuple(zip(*args))

"""
def multi_D_sybmols_to_1D(x):
    '''
    x is an iterable where each element is multidimensional. Change each multidimensional item by an integer key
    '''
    #_pdb.set_trace()
    lookuptable = {}
    output = []
    next_index = 0
    for element in x:
        # dictionary keys have to be immutable, turn element to a tuple
        element = tuple(element)
        if element not in lookuptable:
            lookuptable[element] = next_index
            next_index += 1
        output.append(lookuptable[element])

    return output
"""
def mi(x, y):
    '''
    compute and return the mutual information between x and y
    
    inputs:
    -------
        x, y:   iterables of hashable items
    
    output:
    -------
        mi:     float

    Notes:
    ------
        if you are trying to mix several symbols together as in mi(x, (y0,y1,...)), try
                
        info[p] = _info.mi(x, info.combine_labels(y0, y1, ...) )
    '''
    #_pdb.set_trace()
    # dict.values() returns a view object that has to be converted to a list before being converted to an array
    if isinstance(x, zip):
        x = list(x)
    if isinstance(y, zip):
        y = list(y)

    probX = labels_to_prob(x)
    probY = labels_to_prob(y)
    probXY = labels_to_prob(zip(x, y))

    return getEntropy(probX) + getEntropy(probY) - getEntropy(probXY)

def cond_mi(x, y, z):
    '''
    compute and return the mutual information between x and y given z, I(x, y | z)
    
    inputs:
    -------
        x, y, z:   iterables with discrete symbols
    
    output:
    -------
        mi:     float

    implementation notes:
    ---------------------
        I(x, y | z) = H(x | z) - H(x | y, z)
                    = H(x, z) - H(z) - ( H(x, y, z) - H(y,z) )
                    = H(x, z) + H(y, z) - H(z) - H(x, y, z)
    '''
    #_pdb.set_trace()
    # dict.values() returns a view object that has to be converted to a list before being converted to an array
    probXZ = labels_to_prob(combine_labels(x, z))
    probYZ = labels_to_prob(combine_labels(y, z))
    probXYZ =labels_to_prob(combine_labels(x, y, z))
    probZ = labels_to_prob(z)

    return getEntropy(probXZ) + getEntropy(probYZ) - getEntropy(probXYZ) - getEntropy(probZ)

def mi_chain_rule(X, y):
    '''
    Decompose the information between all X and y according to the chain rule and return all the terms in the chain rule.
    
    Inputs:
    -------
        X:          iterable of iterables. You should be able to compute [mi(x, y) for x in X]

        y:          iterable of symbols

    output:
    -------
        ndarray:    terms of chaing rule

    Implemenation notes:
        I(X; y) = I(x0, x1, ..., xn; y)
                = I(x0; y) + I(x1;y | x0) + I(x2; y | x0, x1) + ... + I(xn; y | x0, x1, ..., xn-1)
    '''
    
    # allocate ndarray output
    chain = _np.zeros(len(X))

    # first term in the expansion is not a conditional information, but the information between the first x and y
    chain[0] = mi(X[0], y)
    
    #_pdb.set_trace()
    for i in range(1, len(X)):
        chain[i] = cond_mi(X[i], y, X[:i])
        
    return chain

def binned(x, binsN, mode, maxX=None, minX=None):
    '''
    bin signal x using 'binsN' bin. If minX, maxX are None, they default to the full range of the signal. If they are not None, everything above maxX gets assigned to binsN-1 and everything below minX gets assigned to 0
    Acutal binning depends on mode:

    mode:   0           bin_size = (maxX-minX)/binsN

            1           bin_size is adaptive to get equal number of responses in each bin.

    '''
    #_pdb.set_trace()
    if maxX is None:
        maxX = x.max()

    if minX is None:
        minX = x.min()

    if mode==0:
        bins = _np.linspace(minX, maxX, binsN)
    elif mode==1:
        percentiles = list(_np.arange(0, 100.1, 100/binsN)) 
        bins = _np.percentile(x, percentiles)
    
    
    # digitize works on 1d array but not nd arrays. So I pass the flattened version of x and then reshape back into x's original shape at the end
    return _np.digitize(x.flatten(), bins).reshape(x.shape)


def gaussianEntropy(var):
    '''
    compute the entropy of a gaussian distribution with variance 'var' using differential entropy formula. Works for covariance matrix as well

    var:    float, the variance
            ndarray, a covariance matrix
    '''
    #_pdb.set_trace()

    
    try:
        # assuming var is a covariance matrix (a ndarray of at least 2 dimensions)
        return 0.5*_np.log2( (2*_np.pi*_np.e)**var.shape[0] * _np.linalg.det(var) )
    except:
        # if it failed, then most likely is just a number, but could be ndarray with just one object
        if _np.iterable(var):
            var = var[0]
        return 0.5*_np.log2(2*_np.pi*_np.e * var)

def gaussianJointEntropy(cov, X):
    '''
    Compute the joint Entropy of the gaussian variables described by 'X' with covariance 'cov'
    
    for example, cov can span 10 different dimensions but I might want to know the entropy of a few dimensions
        gaussianJointEntropy(cov, [2,5])

    inputs:
    -------
        cov:        2D ndarray, the covariance matrix

        X:          iterable of ints, points along the covariance index to use in computing information
                    len(points)>0

        
    outputs:
    --------
        Entropy:    float
    '''

    # limit the covariance to just dimensions described in X
    covX = subCov(cov, X)
    return gaussianEntropy(covX)

def gaussianInformation(cov, X, Y):
    '''
    Compute the mutual information between the gaussian variables described by 'points' with covariance 'cov'
    
    for example, cov can span 10 different dimensions but I might want to know the information between two of those dimensions and a third one as in
        gaussianInformation(cov, [2,5], [6])

    this function computes:
        I(X; Y) = H(X) + H(Y) - H(X;Y)
    
    inputs:
    -------
        cov:        2D ndarray, the covariance matrix

        X/Y:        iterable of ints, points along the covariance index to use in computing information
                    len(points)>0

        
    outputs:
    --------
        information:    float
    '''

    # X+Y is concatenation, if either of them is ndarray will fail. Force them to be lists
    if isinstance(X, _np.ndarray):
        X = list(X)
    if isinstance(Y, _np.ndarray):
        Y = list(Y)

    # extract the submatrices from cov corresponding to X, Y and X+Y
    covX = subCov(cov, X)
    covY = subCov(cov, Y)
    covXY = subCov(cov, X+Y)

    return gaussianEntropy(covX) + gaussianEntropy(covY) - gaussianEntropy(covXY)

def subCov(cov, points):
    '''
    Extract the sub covariance matrix from cov corresponding to 'points'.
    
    inputs:
    -------
        cov:        2D ndarray, a symmetric matrix representing the covariance matrix

        points:     points along the axis of cov to extract the sub matrix
                    Since cov is symmetric, it doesn't matter whether the points refer to dimension 0 or 1 in cov

    outputs:
    --------
        subCov:     2D ndarray, the sub array form cov specified by points.

    usage:
    ------
        subCov(cov, [2,3])

    implementation notes:
        I will take elements from a flatten version of covG. The function to take the elements form a 1D array is 'take' and I'm using product form itertools to get all combinations of the indexes in points. At the end I'm reshaping it to be a square matrix with each dimension having len(points) elements
    '''
    from itertools import product
    return _np.take(cov.flatten(), [i[0]+ cov.shape[0]*i[1] for i in product(points, points)]).reshape(-1, len(points))
    
def ismutable(x):
    '''
    return True if mutable, False if not
    '''
    if isinstance(x, (str, int, float, bool, tuple)):
        return False
    else:
        return True
