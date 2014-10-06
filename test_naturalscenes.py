'''
naturalscenes tests
'''
import nose
import numpy as np
import naturalscenes as ns
import information as info
import pdb

def setup():
    pass

def test_01():
    '''
    Signal is gaussian and white in time and noise is .1 the variance of the gaussian
    MI bewteen noisy and noiseless samples at a point in time should not depend on whether we condition on samples at other times
    '''
    # define some covariance matrices
    covG0 = np.eye(5)
    covN0 = 0.1*covG0
    
    # compute MI between noisy and noiseless versions of the signal
    info0 = ns._getCondInfoP0(covG0, covN0, 0, [],[])

    # since signal is white in time, MI between noisy and noisless versions should not depend on samples at other times
    infoCond1 = ns._getCondInfoP0(covG0, covN0, 0, [1],[])
    infoCond2 = ns._getCondInfoP0(covG0, covN0, 0, [],[1])
    infoCond3 = ns._getCondInfoP0(covG0, covN0, 0, [1],[2])

    nose.tools.assert_almost_equal(info0, infoCond1)
    nose.tools.assert_almost_equal(info0, infoCond2)
    nose.tools.assert_almost_equal(info0, infoCond3)

def test_02():
    '''
    Assume perfectly correlated signals
    Condition on a noiseless signal at a latter time should give the noise entropy
    
    For perfectly correlated signals, H(X,X) = -inf and can't deal with it
    '''
    pass

    """
    covG = np.eye(2)
    covN = .1*covG
    
    info0 = info.gaussianEntropy(covN[1,1])
    infoCond1 = ns._getCondInfoP0(covG, covN, 0, [1],[])
    
    nose.tools.assert_almost_equal(info0, infoCond1)
    """

def test_adaptation():
    adapt_block = ns.adaptation_block('memory_normalization', 2, 0)
    a=np.ones((10000,2))
    a[:,0]*=2
    b = adapt_block.adapt(a)
    
    # not sure if next line will work, but b[0,:] should be almost equal b[1,:]
    nose.tools.assert_almost_equals(b[:,0], b[:,1])

def test_total_information_1(N):
    '''
    generate an array like binned_g and compute the total_information between g and g, result should be close to N (bits)
    '''
    pdb.set_trace()
    g = np.random.random_integers(0,(2**N)-1, (10000,300))
    g = tuple(map(tuple, g.T))

    result = ns.get_total_info_since_t0(g, g, .8)

    return result

def test_total_information_2(N, noise):
    '''
    generate two array like binned_g and letters in get_total_information_since_t0.
    
    One with N bits and the other, idential to the 1st one plus a uniform distribution with noise bits. I think results should be N-noise if noise < N
    '''
    g = np.random.random_integers(0,(2**N)-1, (10000,300))
    L = g + np.random.random_integers(1, noise(10000,300))
    
    g = tuple(map(tuple, g.T))
    L = tuple(map(tuple, L.T))


    result = ns.get_total_info_since_t0(g, L, .8)
    print(N-noise, results, N-noise-result)

    return result
