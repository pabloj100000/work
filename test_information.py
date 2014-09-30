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

def test_cond_mi():
    # information with itself given iteslf should be about 0
    np.testing.assert_approx_equal(0, info.cond_mi(x, x, x), significant=2)
    
    # conditioning on something else doesn't distroy information
    np.testing.assert_approx_equal(bitsN, info.cond_mi(x, x, y), significant=2)
    
    # if no information is there, conditioning on something should not create it
    np.testing.assert_approx_equal(0, info.cond_mi(x, y, y), significant=2)
    np.testing.assert_approx_equal(0, info.cond_mi(x, y, x), significant=2)


