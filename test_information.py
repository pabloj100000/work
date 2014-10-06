'''
test_information
'''

import nose
import numpy as np
import naturalscenes as ns
import information as info

def setup():
    bitsN = 3
    x = np.random.randint(0, 2**bitsN, 100000)
    y = np.random.randint(0, 2**bitsN, 100000)

def test_binned():
    binsN = 8
    x = np.random.randn(1000)
    y = info.binned(x, binsN)
    print(min(y))
    print(max(y))
    assert min(y)==0, 'Error in info.binned. After binning there is a bin assigned less than 0'
    assert max(y)<binsN, 'Error in info.binned. After binning there is a bin assigned more than binsN-1'

def test_binned2():
    binsN=8
    x = np.random.randn(1000)
    y = info.binned(x, binsN, maxX=.5)

    print(min(y))
    print(max(y))
    assert min(y)==0, 'Error in info.binned. After binning there is a bin assigned less than 0'
    assert max(y)<binsN, 'Error in info.binned. After binning there is a bin assigned more than binsN-1'

def test_mi():
    # information with itself has to be about bitsN
    np.testing.assert_approx_equal(bitsN, info.mi(x, x), significant=2)

    # half the numbers are the same, half are mixed, MI should be bitsN/2
    y = np.zeros_like(x)
    y[:len(x)/2] = x[:len(x)/2]
    y[len(x)/2:] = np.random.randint(0, 2**bitsN, len(x)/2)

def test_mi2():
    '''
    Test mi with combined symbols
    '''
    N = 3
    l0 = np.random.random_integers(1,2**N, 100000)
    l1 = np.random.random_integers(1,2**N, 100000)
    l2 = np.random.random_integers(1,2**N, 100000)
    l3 = np.random.random_integers(1,2**N, 100000)

    t0 = info.combine_labels(l0, l0)
    t1 = info.combine_labels(l0, l1)
    t2 = info.combine_labels(l1, l0)
    t3 = info.combine_labels(l0, l2)
    t4 = info.combine_labels(l2, l3)

    print(info.mi(t0, t0), info.mi(t0,t1), info.mi(t1, t2), info.mi(t1, t3), info.mi(t1,t4))

def test_cond_mi():
    # information with itself given iteslf should be about 0
    np.testing.assert_approx_equal(0, info.cond_mi(x, x, x), significant=2)
    
    # conditioning on something else doesn't distroy information
    np.testing.assert_approx_equal(bitsN, info.cond_mi(x, x, y), significant=2)
    
    # if no information is there, conditioning on something should not create it
    np.testing.assert_approx_equal(0, info.cond_mi(x, y, y), significant=2)
    np.testing.assert_approx_equal(0, info.cond_mi(x, y, x), significant=2)


