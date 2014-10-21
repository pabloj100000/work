'''
naturalscenes tests
'''
import nose
import numpy as np
import naturalscenes as ns
import information as info
import pdb
import matplotlib.pyplot as plt

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
    #pdb.set_trace()
    adapt_block = ns.adaptation_block('memory_normalization', 2, .05, 0.01)
    a=np.ones((2,10000))
    a[0,:]*=2
    adapted, tax = adapt_block.adapt(a)
    
    np.testing.assert_allclose(adapted[0,:], adapted[1,:], 1e-2, 0, "test_naturalscenes.test_adaptation: Adapted arrays are not similar to each other")

def test_nl(contrast, thresh):
    bipolar = ns.cell(.005)

    lp = bipolar.sim_central_pathway('gaussian', contrast, length=1000, mean=127)

    bipolar.nl_basal.thresh = thresh
    letters = bipolar.nl_basal.torate(lp)

    nl = ns.compute_nl(lp, letters, 100)
    #vesicles = bipolar.adaptation.adapt(letters)
    
    plt.close('nl')
    fig, ax = plt.subplots(num='nl')
    ax.plot(nl[0], nl[1])

    return nl



def test_temporal_filter():
    # make a kernel that is a delta function at t=0
    new_kernel = np.zeros(100)
    new_kernel[0] = 1
    kernel = ns.filter_block(1, new_kernel, 1)

    # make stim
    stim = ns.fake_noise('gaussian', 10)

    lp = kernel.temporal_filter(stim)

    # First len(new_kernel)-1 points of stim were not properly filtered and are not included in lp
    np.testing.assert_almost_equal(lp, stim[len(new_kernel)-1:])

def test_compute_kernel():

    kernel_pnts = 100
    new_kernel = np.zeros(kernel_pnts)
    new_kernel[0] = 1
    f_block = ns.filter_block(1, new_kernel, 1)

    trials = 1000
    contrast = 10
    mean = 127
    length = trials * kernel_pnts * ns.sim_delta_t
    ker_length = kernel_pnts * ns.sim_delta_t
    stim = ns.fake_noise('gaussian', contrast, length=length, mean=mean)
    
    # simulate linear response
    lp = f_block.temporal_filter(stim)
    stim = stim[kernel_pnts-1:]         # now stim and lp_center are aligned

    # Recompute kernel from the stim and the lp
    new_kernel = ns.compute_kernel(stim, lp, kernel_pnts,10)
    
    plt.close('kernel_test')
    fix, ax = plt.subplots(num='kernel_test')
    ax.plot(f_block.kernel)
    ax.plot(new_kernel)

    return new_kernel, f_block.kernel, stim

def test_fit_birect():
    # make two lines and join them
    x = np.arange(100)
    y = x + 1
    y[60:] = -2*(x[60:]-60) + 61
    best_p, line0, line1, best_fit, error = ns.fit_birect(x,y)

    nose.tools.assert_almost_equal(error, 0)


def run_test_2():
    plt.close('all')
    fix, ax = plt.subplots(nrows=3)

    delta_t= 5      # in ms
    gaussian_psths, nl, constant_psths = test_adaptation_2(delta_t)
    tax = np.arange(0, len(gaussian_psths[0])*delta_t/1000, delta_t/1000)

    for i in range(5):
        ax[0].plot(tax, gaussian_psths[i])
        xticks = [0,.5]
        ax[0].set_xticks(xticks)

    for i in range(1):
        for tw in range(10):
            ax[1].plot(nl[i][tw][0], nl[i][tw][1])
            #xticks = (nl[i][0][0],nl[i][0][-1])
            #ax[1].set_xticks(xticks)
    
    for i in range(4):
        ax[2].plot(tax, constant_psths[i])
        xticks = [0,.5]
        ax[2].set_xticks(xticks)

    return nl

def test_adaptation_2(delta_t):
    '''
    fake gaussian stimuli of different contrasts and pass them through the model
    '''
    #pdb.set_trace()

    mean = 127
    bipolar = ns.cell(delta_t)
    bipolar.adaptation.offset=.6
    bipolar.nl_basal.thresh=59

    # I want to simulate the central pathway for N trials each lasting the same as periphery_kernel
    trials = 1000
    length = trials * len(bipolar.periphery.kernel) * ns.sim_delta_t

    gaussian_psth = []
    nls = []
    for c in [3,6,12,24,100]:
        # simulate central pathway
        lp = bipolar.sim_central_pathway('gaussian', c, length=length, mean=mean)

        # redimension lp to be shape = (trials, len(peirphery.kernel))
        lp = lp.reshape(trials, -1)

        # add peipheral pathway 
        lp += bipolar.periphery.weight*bipolar.periphery.kernel

        # generate [Ca] from lp, using peripheral signal
        letters = bipolar.nl_basal.torate(lp)
        
        # generate vesicle release by adapting [Ca]
        vesicles = bipolar.adaptation.adapt(letters)

        # generate PSTH from vesicles
        psth_pnts = len(bipolar.periphery.kernel)
        gaussian_psth.append(vesicles.mean(axis=0))

        #nl.append(ns.compute_nl(lp.flatten(), vesicles.flatten(), 100))
        # Divide vesicles and lp into time bins according to time from saccade and compute nls
        nl_perTW =[]
        for i in range(10):
            startP = int(i*lp.shape[1]/10)
            endP = int((i+1)*lp.shape[1]/10)
            nl_perTW.append(ns.compute_nl(lp[:,startP:endP].flatten(), vesicles[:,startP:endP].flatten(),100))

        nls.append(nl_perTW)

    constant_psth = []
    trials = 10
    length = trials * len(bipolar.periphery.kernel) * ns.sim_delta_t
    for m in [32,64,128,256]:
        # simulate central pathway
        lp = bipolar.sim_central_pathway('gaussian', 0, length=length, mean=m)

        # redimension lp to be shape = (trials, len(peirphery.kernel))
        lp = lp.reshape(trials, -1)

        # add peipheral pathway 
        lp += bipolar.periphery.weight*bipolar.periphery.kernel

        # generate [Ca] from lp, using peripheral signal
        letters = bipolar.nl_basal.torate(lp)
        
        # generate vesicle release by adapting [Ca]
        vesicles = bipolar.adaptation.adapt(letters)

        # generate PSTH from vesicles
        psth_pnts = len(bipolar.periphery.kernel)
        constant_psth.append(vesicles.mean(axis=0))
        
    return gaussian_psth, nls, constant_psth


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
