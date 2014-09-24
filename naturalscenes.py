'''
naturalscenes.py

A module to load and process natural scenes from Tkacik database
'''
import numpy as _np
from scipy import ndimage as _nd
from glob import glob as _glob
from itertools import product as _product
import matplotlib.pyplot as _plt
import information as _info
import pdb as _pdb
import pandas as _pd
from time import time as _time
import pickle as _pickle
import natural_scenes_fitting as _nsf


# define simulation parameters
pixperdegree = 46       # converts degrees to pixels in image
sim_delta_t = .005           # time resolution of kernels in seconds
rw_step = .01            # in degrees
saccade_size = 6         # in degrees
sim_start_t = -.5            # in seconds
sim_end_t = 1                # in seconds
g = None
images_list = None
tax = None

# define parameters for analyzing words
letter_length = .100
lettersN = 2
binsN = 4
bin_rate = None

# define center pathway parameters
center_size = 1         # in degrees
center_kernel_file = '/Users/jadz/Documents/Notebook/Experiments/Simulations/Natural Images DB/New Saccade analysis/center_kernel.txt'
center_weight = 1

# define surround pathway parameters
surround_size = 2.5     # in degrees
surround_kernel_file = '/Users/jadz/Documents/Notebook/Experiments/Simulations/Natural Images DB/New Saccade analysis/surround_kernel.txt'
surround_weight = .9

# define peripheral pathway parameters, if you change the parameters defining peripheral response, run "generate_peripheral_kernel()"
periphery_size = 0      # "0" mean no spatial integration for this pathway
periphery_kernel_file = '/Users/jadz/Documents/Notebook/Experiments/Simulations/Natural Images DB/New Saccade analysis/peripheral_kernel.txt'
periphery_weight = 1    # overall weight factor for periphery, doesn't affect shape
periphery_exc = 1       # this controlls the relative height of the gating excitatory shift
periphery_inh = -1      # this controlls the relative height of the gating inhibitory window
gating_start_t = .08    # this controlls where gating starts
gating_end_t = .13      # this controlls where gating ends/inhibition starts
recovery_start_t = 0.3  # this controlls where inhibition ends and starts to recover to basal state
recovery_end_t = 0.45   # this controlls when the system returns to basal state

# define parameters for internal threshold
nl_type = 'birect'
nl_threshold = 0
nl_units = 'linear prediction'

# define parameters for adaptation block
adaptation_type = 'memory_normalization'
adaptation_memory = 2           # in seconds


def data_summary():
    '''
    Do everything needed to generate the figures that will go into the paper. Modify as needed
    
    '''
    global g, bin_rate
    
    # generate a bipolar cell object.
    # It has three pathways, center, surround and periphery, each one can contribute to the membrane potential (mp).
    # Then I can add noise to the noiseless mp that is consistent with Yusuf's intracellular recordings
    # Pass that noisy mp through a nonlinearity representing [Ca] concentration
    # Pass [Ca] through an adaptive block to represent vesicle release

    bipolar = cell()

    print('Loading or computing g')
    bipolar.processAllImages()              # this will take several hours unless it is loading from a file
    
    # load a dataframe exported from Igor with Yusuf's data. I also compute a few new columns to associate the simulation with the experiment and be able to estimate the noise in the simulation
    # columns of interest are sim_mp_sd (simulated standard deviation of the membrane potential) and sim_noise (the standard deviation of the membrane potential at the given contrast)
    sim_noise_fit, df = reproduce_Yusuf()

    

    # compute the covariance. I'll use the covariance both for estimating noise (from Yusuf's data) and to compute Shannon's Mutual Informaiton assuming gaussianity
    print('Computing covG')
    covG = _np.cov(g, rowvar=0)
    covG.tofile('Data/LinearPredictionCov')

    # adding noise that is uncorrelated in time to the simulation, is equivalent to adding a diagonal matrix (covN N:noise) to the LP covariance (covG), the diagonal values in covN are the variance of the noise at each point int time. I'm using Yusuf's data to pick at each point in time noise of the appropriate variance
    covN = getNoiseCovariance(covG, sim_noise_fit, .05)
    
    # compare how new information changes when you incorporate more than one letter
    print('Computing cond information with several letters')
    for _lettersN in [1,2,3]:
        info_gained_by_last_letter(covG, covN, letter_length, _lettersN)
    
    # compare how new information changes as a funciton of letter size
    print('Computing cond information as a function of letter size')
    for _letter_length in [.005, .025, .125]:
        info_gained_by_last_letter(covG, covN, _letter_length, lettersN)
    
    # compare the amount of information obtained when the last letter gates or doesn't gate.
    print('Computing the difference between gating and no gating')
    wrap_gating_effect(g, letter_length, nl, binsN)
    
    # Generate data to show the transition to a gating-nl that lasts longer than a single letter (two letters).
    #   if word does not overlap the gating window, information is as in the non-gating mode
    #   if word straddles the edges of the gating window, it has one letter with each nonlinearity
    #   if word is embeded in gating window, both letters are from gating nl
    print('Computing total information with gating and no gating')
    get_word_info(g, gating_start_t, gating_end_t, letter_length, lettersN, nl, binsN)
    
    # make all plots
    plot_summary()

def plot_summary():
    global g, bin_rate, letter_length, lettersN, binsN, gating_start_t, gating_end_t
    
    if g is None: 
        g = _np.fromfile('Data/LinearPrediction').reshape(-1,300)

    sampleG(g, 100)
    
    nl = nonlinear_block(nl_s_type, nl_thresh, nl_units)
    
    #_pdb.set_trace()
    generate_periphery_kernel()
    fig, ax = plot_compare_letters_N(int(letter_length*1000), lettersN_list=[1, 2,3], nameout='Compare Letters N A')
    fig, ax = plot_compare_letters_N(int(letter_length*1000), lettersN_list=[2,3], nameout='Compare Letters N B')
    fig, ax = plot_compare_letters_N(int(letter_length*1000), lettersN_list=[1], nameout='Compare Letters N C', )
    ax.set_ylim(1.25, 1.75)
    fig.canvas.draw()
    fig.savefig('Figures/Compare Letters N C.pdf')


    plot_compare_letter_length(lettersN, length_list=[5,25,125])

    plot_gating_effect(letter_length)

    plot_word_info([2], [int(letter_length*1000)])

    explore_word_letters(g, gating_start_t, gating_end_t, nl, bin_rate)


def load_UFlicker_PSTH(i):
    '''
    load all files of the form UFlicker_PSTH_'i'c_#c and concatenate them together. If file doesn't exist returns None

    Choice of file name is not the best but 1st #c refers to igor's point in UFlikcer Summary experiment wave :allCells:w_mask
        If w_mask[k] is set to 1 then I have exported all PSTHs associated with that wave and name will be UFlicker_PSTH_'k'c_#c

    Second #c is the contrast.
    
    output:
    -------
        if file exists:
            psth (1d ndarray):      one long psth with all conditions one after the other

            psth_pnts (int):        number of points in a given PSTH

        if file doesn't exist:
            None
    '''

    from os import listdir

    #_pdb.set_trace()
    files = [name for name in listdir('UFlicker PSTHs') if name.startswith('UFlicker_PSTH_c{0}_'.format(i))]

    if files == []:
        return None

    files.sort(key=lambda x: int(x.split('_')[3][:-5]))
    
    psths = [_np.fromfile('UFlicker PSTHs/'+one_file, sep='\n') for one_file in files]

    x = _np.arange(0, len(psths[0])*len(files)*sim_delta_t, sim_delta_t)
    return x, _np.concatenate(psths), len(psths[0])

def load_StableObject_PSTH(psth_pnts=None):
    '''
    load psths from file StableObject_PSTH.txt, separating saccades from still.

    if psth_pnts is given then the number of samples of each psth is adjusted to be psth_pnts

    output:
    -------
        sacc_psths (1d ndarray):     one long psth with all saccading conditions one after the other

        still_psths (1d ndarray):    one long psth with all still conditions one afte the other

        psth_pnts (int):        number of points in a given PSTH
    '''

    from os import listdir
    from scipy.signal import resample

    psths = _np.fromfile('StableObject_PSTH.txt', sep='\n').reshape(-1, order = 'F')
    lumN = 4

    still_psths = psths[:len(psths)/2]
    sacc_psths = psths[len(psths)/2:]

    x = _np.linspace(0, lumN*.5, len(still_psths))
   
    #_pdb.set_trace()
    if psth_pnts is not None:
        still_psths = resample(still_psths, lumN*psth_pnts)
        sacc_psths = resample(sacc_psths, lumN*psth_pnts)
        x = _np.linspace(0, lumN*.5, lumN*psth_pnts)


    _plt.close('StablePSTHs')
    fig, ax = _plt.subplots(num='StablePSTHs')

    colors = 'krgb'
    ax.plot(x, still_psths)
    ax.plot(x, sacc_psths, ':')

    return x, still_psths.flatten(), sacc_psths.flatten()

def load_TNF_PSTHs():
    # load TNF PSTHs that were exported from igor. PSTHs for still/saccade conditions are stored in TNF_still/saccade_PSTHs.txt.
    # Each file has psth for all cells. Each PSTH has 96 pnts spaced every .005s
    still_psth = _np.fromfile('TNF_still_PSTHs.txt', sep='\t').reshape(-1, 96)
    sacc_psth = _np.fromfile('TNF_saccade_PSTHs.txt', sep='\t').reshape(-1, 96)
    tax = _np.arange(0, 96*.005, .005)

    return still_psth, sacc_psth, tax


def _getImagesPath(path=None):
    if path is None:
        path = '/Users/jadz/Documents/Notebook/Matlab/Natural Images DB/RawData/*/*LUM.mat'
        
    global images_list
    images_list = _glob(path)

def _loadImage(imNumber):
    '''
    Load an image from the database. Image undergoes light adaptation. THe mean of the image is forced to be 127.

    inputs:
    -------
        imNumber:   integer, specifying which element from images_list to load

    output:
        image:      ndarray with the image
    '''
    from scipy import io
    global images_list

    if images_list is None:
        _getImagesPath()

    # load matlab array 
    image = io.loadmat(images_list[imNumber])['LUM_Image']

    # perform light adaptation
    image *= 127/image.mean()

    return image

def _getEyeSeq(filter_length):
    '''
    Generate a sequence of eye movements in both x and y directions
    The sequence is a 2D ndarray compossed of steps. 
    seq[0][p] is the step in the x direction at point p
    seq[1][p] is the step in the y direction at point p

    seq starts at time sim_start_t - ( len(filter_instance.center_kernel) - 1 ) * sim_delta_t and ends at time sim_end_t
    in this way, when convolving with mode='valid' the output will have samples spanning sim_start_t and sim_end_t

    intput:
    -------
        filter_length:        number of points of the filter that will be used in convolving the time sequence. filter_length -1 points are needed in front of the eye movement sequence such that the convolution with the filter will have the right number of points when using 'valid'

    output:
    -------
        seq:    2D ndarray with steps
    '''

    stepsN = int((sim_end_t-sim_start_t)/sim_delta_t + filter_length - 1)

    # generate the FEM part of the sequence
    seq = _np.random.randn(2, stepsN)
    seq *= pixperdegree*rw_step

    # add saccade in both x and y for the time being. The distribution of LP I'm getting is skewed to the right as if most images were transitioning from light to dark patches.
# I think this might be due to the fact that I'm always saccading in the same direction (may be from sky to dirt). I will randomize here the direction of the saccade but keeping both fixational points the same.
    saccadePnt = int(filter_length - 1 - sim_start_t/sim_delta_t)
    if _np.random.rand() > 0.5:
        # saccade in the usual way
        seq[:,saccadePnt] += saccade_size*pixperdegree
    else:
        # saccade backwards
        seq[:,0] += saccade_size*pixperdegree
        seq[:,saccadePnt] -= saccade_size*pixperdegree

    # change from steps to actual positions
    seq = seq.cumsum(1)

    return seq.astype('int16')


def _getTAX():
    global tax
    if tax is None:
        tax = _np.arange(sim_start_t, sim_end_t, sim_delta_t)

    return tax
    

def sampleG(g, num):
    '''
    make a plot with 'num' random cells
    '''
    #_pdb.set_trace()
    global tax
    if tax is None:
        tax = _getTAX()
    
    _plt.close('Sample G')
    fig, ax = _plt.subplots(num='Sample G')
    
    # add 'num' traces to plot. Traces are chosen randomly from 1st dimension of g
    traces = []
    for i in range(num):
        index = _np.random.randint(0, g.shape[0])
        traces.append(ax.plot(tax, g[index,:]))

    # add dash line at saccade
    traces.append(ax.plot((0,0), (-5,5), 'k:'))

    # add labels
    ax.set_xlabel('Time (s)', fontsize=10)
    #ax.set_ylabel('Linear Prediction (AU)')
    ax.set_ylabel('Filtered\nlight (AU)', fontsize=10)
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # adjust bottom margin so that label is shown
    fig.subplots_adjust(left=.25, bottom=.35)

    ax.set_xticks((-.2, 0, .2, .4, .6))
    ax.set_xticklabels((-.2, 0, .2, .4, .6), color='k', fontsize=10)
    ax.set_yticks((-5, 0, 5))
    ax.set_xlim(-.25, .75)
    ax.set_yticklabels([])
    #ax.yaxis.set_visible(False)

    #fig.set_size_inches(2, 1.5)
    fig.savefig('Figures/samplesG.pdf', transparent=True)
    return fig, ax, traces

def information(cov, X):
    '''
    cov is the covariance matrix of the simulation. Each point along either of the two axis represents a point in time (from sim_start_t to sim_end_t in steps of length sim_delta_t)
    Here, produce a 1D ndarray that at each point 'p' (corresponding to time startT + p*deltaT) , computes the mutual information between the linear prediction at points X (relative to each p) and p.
    Coputes MI(X+p, p) and here the +P is not list concatenation but item by item summation
    For example: information(cov, [-1, -2, -3]), computes for every point p: infomration(cov, [-1+p, -2+p, -3+p], p)
    
    inputs:
    -------
        cov:    2D ndarray, covariance matrix of the linear prediction

        X:      iterable of ints, points relative to each point in the time axis to compute MI with

    outputs:
    --------
        information     1D ndarray, each point has the information between that point and X
                        size of output is same as cov.shape[0]
                        
    '''

    # first lets convert X to ndarray it it is not
    if not isinstance(X, _np.ndarray):
        X = _np.array(X, dtype='int')
    
    return _np.array([_info.gaussianInformation(cov, X+p, [p]) if p+X.min()>=0 and p+X.max()<cov.shape[0] else _np.nan for p in range(cov.shape[0])])
    #return [(X+p, p) if p+X.min()>=0 and p+X.max()<cov.shape[0] else _np.nan for p in range(cov.shape[0])]

def normalize(g, sub_g):
    '''
    I'm going to normalize all values of g, using the mean and std coming from sub_g
    
    in this way I can have each point in time be normalized according to the present mean and contrast or I can normalize to a different mean and contrast (pehaps what happend 100ms ago) 
    '''

    g -= sub_g.mean()
    g /= sub_g.std()

    return g
"""
def getCondInfo(cov, covN, condListLP, condListLPN):
    '''
    wrapper to all _getCondInfoP0 on all points p0 of cov
    '''

    #_pdb.set_trace()
    allPoints = condListLP + condListLPN
    if len(allPoints)==0:
        minP=0
        maxP=0
    else:
        minP = min(allPoints)
        maxP = max(allPoints)

    # preallocat information
    info = _np.zeros(cov.shape[0])

    for p0 in range(cov.shape[0]):
        if p0+minP>=0 and p0+maxP<cov.shape[0]:
            noiselessPoints = [p0 + p for p in condListLP]
            noisyPoints = [p0 + p for p in condListLPN]

            info[p0] = _getCondInfoP0(cov, covN, p0, noiselessPoints, noisyPoints)
        else:
            info[p0] = _np.nan

    return info
"""
def _getCondInfoP0(cov, covN, p0, condListLP, condListLPN):
    '''
    Compute the conditional information between the LP and the LP + noise at time corresponding to point p0, conditional on all time points from condListLP and condListLPN

    Computes:
        I(g(t); g(t)+n | g(ta_0), ..., g(ta_n), g(tb_0)+noise, ..., g(tb_n)+noise)

        where ta_0, ... ,ta_n are times corresponding to points in condListLP and tb_0, ..., tb_n are times corresponding to points in condListLPN

    inputs:
    -------
        cov:    2D ndarray, covariance matrix of the linear prediction, comres from the simulation

        covN:   2D ndarray, covariance matrix of the noise, comes from the variance in the simulation and Yusuf's intracellular data

        p0:     int, point corresonding to time in g(t) and g(t)+n in the calculation
                p0 = (t - sim_start_t)/sim_delta_t

        condListLP:     list of ints. Point is relative to p0. -1 represents sim_delta_t prior to p0, etc.
                        allows to condition the information calculation at time t on g(t0), g(t1), etc whrere t0, t1, correspond to points in condListLP
        
        condListLPN:    idem condListLP but conditions on g(t0)+noise for all t0 in condListLPN

    output:
    -------
        info:   1D ndarray, the conditional mutual information

    Implemenation notes:
        I(X; Y | Z) = H(X | Z) - H(X | Y, Z)
                    = H(X, Z) - H(Z) - ( H(X, Y, Z) - H(Y, Z) )
                    = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)


        In computing all these entropies, I will generate the covariance matrix between LP and LP + Noise for the time points requested. The covariance between different time pionts of LP is just a submatrix of LP. The covariance matrix between different time points of LP + noise is a submatrix of covarianceLP + the corresponding diagonal terms from covN. The covariance matrix between g(t0) and g(t1)+N is a submatrix of cov (at t0, and t1) with noise from covN added to t1
    '''
    if not isinstance(condListLP, list):
        condListLP = list(condListLP)
    if not isinstance(condListLPN, list):
        condListLPN = list(condListLPN)

    # According to the implementaion note, I will need to compute 4 different entropies. Extract the subCov corresponding to each one of them
    XZ = _extractSubCov(cov, covN, [p0] + condListLP, condListLPN)
    YZ = _extractSubCov(cov, covN, condListLP, [p0] + condListLPN)
    XYZ = _extractSubCov(cov, covN, [p0] + condListLP, [p0] + condListLPN)
    Z = _extractSubCov(cov, covN, condListLP, condListLPN)
    
    if len(Z)==0:
        # Special case, only used if not conditioning on anything, it actually computes the I(X,Y) sinze Z is empty
        return _info.gaussianEntropy(XZ) + _info.gaussianEntropy(YZ) - _info.gaussianEntropy(XYZ)
    else:
        return _info.gaussianEntropy(XZ) + _info.gaussianEntropy(YZ) - _info.gaussianEntropy(XYZ) - _info.gaussianEntropy(Z)

def _extractSubCov(covLP, covN, noiselessPoints, noisyPoints):
    '''
    Given the covariance for the LP and the covariance of the noise, extract a new covariance matrix that corresponds to noiseless and noisyPoints
    
    Assume that there are 'A' noiseless points and 'B' noisy points.

    The output is a covariance matrix of dimension (A+B) x (A+B), the interaction among points in this submatrix might be due to correlations in the linear prediction or in the noise.

    inputs:
    -------
        covLP:  2D ndarray, covariance matrix of the linear prediction, comes from the simulation

        covN:   2D ndarray, covariance matrix of the noise, comes from the variance in the simulation and Yusuf's intracellular data

        noiselessPoints:     list of ints. Each piont corresponds to g(t) through p=(t-sim_start_t)/sim_delta_t
        
        noisyPoints:         list of ints. Each piont corresponds to g(t)+noise through p=(t-sim_start_t)/sim_delta_t

    output:
    -------
        subCov:   2D ndarray, the covariance matrix of the points choosen

        I will generate the covariance matrix between LP and LP + Noise for the time points requested. The covariance between different time pionts of LP is just a submatrix of LP. The covariance matrix between different time points of LP + noise is a submatrix of covarianceLP + the corresponding diagonal terms from covN. The covariance matrix between g(t0) and g(t1)+N is a submatrix of cov (at t0, and t1) with noise from covN added to t1 and no off diagonal element because I'm assuming that noise is uncorrelated in time.
    '''

    allPoints = noiselessPoints + noisyPoints

    if allPoints==[]:
        return _np.array([])

    # extract the subarray corresponding to allPoints from covLP. I'm taking points with 'take' from a flatten version of cov. At this point in the code, there is no reference to the noise
    from itertools import product
    subCovG = _np.array([_np.take(covLP.flatten(), coord[0]+covLP.shape[0]*coord[1]) for coord in product(allPoints, allPoints)]).reshape(-1, len(allPoints))
    #print(subCov)
    
    # In delaing with noisy point I replace all noiseless points by None such that testing becomes simple. Now after product if any of the points comes from noiselessPoints, the test "None in coord" will return True
    allPoints = [None]*len(noiselessPoints) + noisyPoints
    subCovN = _np.array([0 if None in coord else _np.take(covN.flatten(), coord[0]+covN.shape[0]*coord[1]) for coord in product(allPoints, allPoints)]).reshape(-1, len(allPoints))
    
    return subCovG + subCovN
    '''
    # add noise in the diagonal terms corresponding to noisyPoints. Since I'm assuming that different time points have noise that is uncorrelated, then there are no contributions to off diagonal terms, unless there is a point more than once in noisyPoints. In that case, the contribution to the off diagonal term is covN[p,p]
    #_pdb.set_trace()
    N0 = len(noiselessPoints)   # all N0 first dimensions of subCov correspond to noiseless points
    N1 = len(noisyPoints)       # then, there are N1 dimesnions with noise.

    for i, coords in enumerate(product(noisyPoints, noisyPoints)):
        if coords[0]==coords[1]:
            i0 = _np.mod(i, N1)+N0
            i1 = _np.floor(i/N1)+N0
            subCov[i0, i1] += covN[coords]
    '''

def getInfo0(covG, covN):
    '''
    compute the mutual information between a gaussian process with covariance covG and a noisy version corrupted by an additive gaussian process with covariance covN
    I( g ; g+noise) = H(g + noise) - H(g + noise | g)
                    because g  and noise are both gaussian and uncorrelated this ends up being:
                    = 0.5 * log2( 2*pi*e * (varG + varN) ) - 0.5 * log2(2*pi*e * varN)
                    = 0.5 * log2( (varG + varN)/varN)
                    = 0.5 * log2( 1 + SNR )
                    which is a well known result

    '''
    return _np.array( [0.5 * _np.log2(1 + covG[i,i]/covN[i,i]) for i in range(covG.shape[0])])
    
def getNoiseCovariance(covG, sim_noise_fit, decay_time):
    '''
    ******** Very Important *********
    * Everything is in the simulation units and not in Yusuf's units.
    *********************************
    
    From Yusuf's data, I have the noise in the Bipolar cell's membrane potential as a function of contrast (stim SD)
    I have reproduce Yusuf experiments in reproduce_Yusuf() and scaled the noise to be in simulation units.

    Here I will return the variance of the noise, given the variance in the signal
    
    Implementation Notes:
        I'm implementing noise that is correlated in time and decays exponentialy. therefore in order to compute the noise I have to do two things
        1. The diagonal elements of the noise are coming streight from Yusuf's data:
        2. The off diagonal terms are a mixture of the diagonal term noise that decays exponentialy with time.

        In computing diagonal terms of noise from yusuf data:
            covG[i,i] is the variance at point i.
            convert to SD
            use the linear fit to convert the stimulus SD into a noise SD
            convert back to a variance

    inptus:
    -------
        covG (2d ndarray):    comes from passing images that are in the range 0-255 through some filter and then computing cov.

        sim_noise_fit (poly1d object):      linear fit to sim_nosie_sd vs sim_mp_sd

        decay_time:     time pionts t0 and t1 have noise that is correlated according to exp(-abs(t0-t1)/decay_time)


    output:
    -------
        noise:  noise the cell would experience under such input variance
    '''
    #_pdb.set_trace()

    # if covG is a number, just compute the noise and return it
    if not _np.iterable(covG):
        return sim_noise_fit(_np.sqrt(covG))**2

    # generate the covariance matrix for the noise. same shape as covG
    covN = _np.zeros_like(covG)

    # the diagonal values are computed from Yusuf's intracellular data
    for i in range(covG.shape[0]):
        covN[i,i] = sim_noise_fit(_np.sqrt(covG[i,i]))**2

    # To speed things I'm first computing the delay in points beyond which the correlation is too small (in those cases I will ignore it). Then I'll fill covN around the diagnol (skipping diagonal terms for which I already computed the noise

    max_distance = int(-_np.log(.1)*decay_time/sim_delta_t)     #log is the natural logarithm
                                                                # .1 is a hardcoded constant signaling when exp(-t/tau) = 0.1
    #_pdb.set_trace()
    for j in range(covG.shape[0]):
        for dist in range(1, max_distance):
            if j+dist>= covG.shape[0]:
                continue

            covN[j+dist, j] = _np.sqrt(covN[j+dist,j+dist]*covN[j,j])*_np.exp(-dist*sim_delta_t/decay_time)
            covN[j, j+dist] = covN[j+dist, j]
    
    return covN

def generate_peripheral_kernel(points):
    '''
    points should be the deisred number of points for the kernel, probably the same as in the central/surround kernels to avoid problems
    '''

    #_pdb.set_trace()
    kernel = _np.zeros(points)

    # convert gating_start/end_t to points and set the values of the kernel to periphery_exc
    gating_start_p  = int(gating_start_t/sim_delta_t)
    gating_end_p    = int(gating_end_t/sim_delta_t)
    kernel[gating_start_p:gating_end_p] = periphery_exc

    # now set all points in between gating_end_p and recovery_start_t to periphery_inh
    recovery_start_p = int(recovery_start_t/sim_delta_t)
    kernel[gating_end_p:recovery_start_p] = periphery_inh

    # now set all points in between recvoery_start/end_t to a line joining periphery_inh and 0
    recovery_end_p = int(recovery_end_t/sim_delta_t)
    kernel[recovery_start_p:recovery_end_p] = _np.arange(periphery_inh, 0, (0-periphery_inh)/(recovery_end_p-recovery_start_p))

    kernel *= periphery_weight

    kernel.tofile(periphery_kernel_file, sep=' ')


def time_to_point(t, return_flag):
    '''
    convert time to the nearest point in the time axis
    
    inputs:
    -------
        t (float):              in seconds

        return_flag (int):      0, return point such that time at the point is less than t
                                1, round to the nearest point
    '''
    p = (t-sim_start_t)/sim_delta_t
    if return_flag:
        return int(p)
    else:
        return int(_np.floor(p))

def point_to_s(point):
    '''
    convert from point in covG or tax to time

    input:
    ------
        point (int):    the point to get the corresponding time of

    output:
    -------
        time (int):     time in s
    '''
    return _getTAX()[point]

def chain_rule_info(covG, covN, letter_length, lettersN):
    '''
    wrapper to call _chain_rule_info(covG, covN, p0, points) on every p0
    '''
    #_pdb.set_trace()
    
    # preallocate output
    tax = _getTAX()
    chain = _np.zeros((len(tax), lettersN))

    points_per_letter = int(letter_length/sim_delta_t)
    
    for p in range(points_per_letter*(lettersN-1), len(tax)):
        points = [-i*points_per_letter+p for i in range(lettersN-1, -1, -1)]
        chain[p,:] = _chain_rule_info(covG, covN, p, points).T
    
    return chain

def _chain_rule_info(covG, covN, p0, points):
    '''
    Decompose the information according to the chain rule and return all the terms in the chain rule.
    
    The information I'm computing is:   I(g(p0) ; g(p1)+n, ..., g(pn)      where p1, p2, ..., pn are in points

    And the chain rule is I(x0, x1, ..., xn; y) = I(x0; y) + I(x1;y | x0) + I(x2;y | x0, x1) + ... + I(xn; y | x0, x1, ..., x(n-1))
    
    Implemenation notes:
        I(X; Y | Z) = H(X | Z) - H(X | Y, Z)
                    = H(X, Z) - H(Z) - ( H(X, Y, Z) - H(Y, Z) )
                    = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
    '''
    
    # allocate ndarray output
    condInfo = _np.zeros(len(points))

    # first term in the expansion is not a conditional information, but the information between the stimulus at point p0 and the 1st noisy measurement from points
    condInfo[0] = _info.gaussianInformation(_extractSubCov(covG, covN, [p0], [points[0]]), [0], [1])
    
    #_pdb.set_trace()
    for i in range(1, len(points)):
        # According to the implementaion note, I will need to compute 4 different entropies. Extract the subCov corresponding for each one of them
        # X is always the noisy contribution at points[i]
        # Y is always the noiseless contribution at p0
        # Z are the noisy points in 0:i (not counting i), therefore Z = points[:i]
        XZ = _extractSubCov(covG, covN, [], points[:i+1])
        YZ = _extractSubCov(covG, covN, [p0], points[:i])
        XYZ = _extractSubCov(covG, covN, [p0], points[:i+1])
        Z = _extractSubCov(covG, covN, [], points[:i])
        
        condInfo[i] = _info.gaussianEntropy(XZ) + _info.gaussianEntropy(YZ) - _info.gaussianEntropy(XYZ) - _info.gaussianEntropy(Z)
        
    return condInfo

def newInformation(covG, covN, letterLength):
    '''
    compute the mutual information between noiseless sample at point p0 and the noisy sample at point p0 conditioning on all previous noisy samples


    implementation notes:
        for each point p0, compute I(g(tn); t(tn)+n | g(t0)+n, g(t1)+n, ..., g(tn)+n)

        I(X; Y | Z) = H(X | Z) - H(X | Y, Z)
                    = H(X, Z) - H(Z) - ( H(X, Y, Z) - H(Y, Z) )
                    = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
    
    where X: noiseless measurement at point pn
    Y:       noisy measurement at point pn
    Z:       noisy measurement at all points prior to pn

    in this case H(Y, Zn) = H(Z(n+1))

    and H(X, Y, Zn) = H(X, Z(n+1))

    Therefore it is faster to first compute all timepoints of both type of entropies and then combine them
    '''
    
    #_pdb.set_trace()

    Zentropy = _np.zeros(covG.shape[0])
    XZentropy = _np.zeros(covG.shape[0])
    
    for p in range(covG.shape[0]):
        Zentropy[p] = _info.gaussianEntropy(_extractSubCov(covG, covN, [], list(range(p+1))))
        XZentropy = _info.gaussianEntropy(_extractSubCov(covG, covN, [p], list(range(p+1))))

def plot_firing_rate_hist(firingRate):
    '''
    firingRate is a 2D ndarray such that firingRate[:,i] is the firing rate in a given condition.
    Histogram all firing rates in 'firingRate', to get an idea of the STD and see which one carries more entropy
    '''

    _plt.close('firingRate_std')
    fig, ax = subplots(num='firingRate_std')
    
    traces = []
    for i in range(firingRate.shape[1]):
        std = firingRate[:,i].std()
        traces.append(_plt.hist(firingRate[:, i], label=r'$\sigma={0}$'.format(std)))

    _plt.legend()
    return fig, ax, traces

def _get_letters(t0, g, covG, nl, binsN=None):
    '''
    I have the linear prediction g as a function of time.
    I'm going to compute FRs or Ca concentrations from g at t0
    In computing the FR or Ca concentration I'm going to use the nl method torate
    
    inputs:
    -------
        t0 (float):         time of letter

        g/covG:             same as always

        nl:                 a nonlinear object

        binsN (int or None):
            if given bins responses

    output:
    -------
        letters (2D ndarray):       letter for each cell in g corresponding to t0

    '''
    #_pdb.set_trace()
    # compute the binned firing rate at both t0 and t0-deltaT for each sigmoid in sigmoids
    if binsN is not None:
        return _info.binned(nl.torate(g[:, time_to_point(t0, 0)]), binsN)
    else:
        return nl.torate(g[:, time_to_point(t0, 0)])

def _get_words(letter_times, gating_start_t, gating_end_t, g, covG, nogating_nl, gating_nl, gating_flag, binsN=None):
    '''
    form words by pasting together letters. 
    Gating starts and ends at times described by gating_start/end_t, outside these times, both gating and no gating use the same letters and words
    Words will be identical if theyn don't overlap with the gating window.
    
    inputs:
    -------
        letter_times (iterable of floats):
            all the times that make a word, for example: [-1.02, -1.00] for a two letter word with last letter 1s before the saccade
                                                         [0.05, .07, .09] for a three letter word

        gating_start_t (float):

        gating_end_t (float):

        g (2D ndarray):             the linear predictions for all cells

        gating_nl:                  nonlinear_block object

        gating_flag (bool):         if True and letter_times[-1] in between gating_start_t and gating_end_t uses gating_sig
                                    if False uses nogating_sig

        binsN (int):                responses are discretized from 0 to binsN-1

    outputs:
    --------
        gating_words (2D ndarray):  for each cell, returns all letters at the requested 'letter_times' times using gating sigmoid where it corresonds.
    '''

    #_pdb.set_trace()
    # allocate memory for gating_word
    words = _np.zeros((g.shape[0], len(letter_times)))
    
    # Compute maximum FR when nonlinearities are just a rectification (assuming slope of 1)
        # maxFR comes by computing max g and then subtracting the threshold
    points = [time_to_point(t) for t in letter_times]
    max_g = g[:, points].max()
    max_fr = max_g - min(nogating_sig[0], gating_sig[0])
    
    nogating_binsN = binsN * max_fr/(max_g-nogating_sig[0])
    gating_binsN = binsN * max_fr/(max_g-gating_sig[0])

    #max_fr = max(gating_sig[0]+gating_sig[1], nogating_sig[0]+nogating_sig[1])
    #nogating_binsN =  round(binsN*(nogating_sig[0]+nogating_sig[1])/max_fr)
    #gating_binsN = round(binsN*(gating_sig[0]+gating_sig[1])/max_fr)
    print('gating_binsN = {0}, nogating_binsN = {1}'.format(gating_binsN, nogating_binsN))
    for i, time in enumerate(letter_times):
        point = time_to_point(time, 0)

        if gating_flag and gating_start_t <= time and time < gating_end_t:
            words[:,i] = gating_nl.torate(g[:,point])
            binsN = gating_binsN
        else:
            words[:,i] = nogating_nl.torate(g[:,point])
            binsN = nogating_binsN

    words[:,i] = _info.binned(words, binsN)

    return words

def get_word_cond_info(g,nls):
    ''' 
    wrapper to call _get_word_cond_info

    input:
    ------
        g

        nls:        iterable of nonlinear_block objects
    
    output:
    -------
        nogating_info

        nogating_tax

        nogating_rate

        gating_info

        gating_tax

        gating_rate

        generates and saves plot Figures/gating_vs_FEM_Word_cond_info
    '''
    #_pdb.set_trace()
    letter_length = .02
    gating_start_t = .05
    gating_end_t = .15
    binsN = 32

    nogating_sig = sigmoids[0]
    gating_sig = sigmoids[1]

    tax = _np.arange(-.25, .55, letter_length)
    if nogating_sig[3]==0:
        nogating_info, nogating_rate = _np.zeros_like(tax), _np.zeros_like(tax)
    else:
        nogating_info, nogating_rate = _get_word_cond_info(letter_length, tax, gating_start_t, gating_end_t, g, covG, nogating_sig, gating_sig, binsN, 0)

    if gating_sig[3]==0:
        gating_info, gating_rate = _np.zeros_like(tax), _np.zeros_like(tax)
    else:
        gating_info, gating_rate = _get_word_cond_info(letter_length, tax, gating_start_t, gating_end_t, g, covG, nogating_sig, gating_sig, binsN, 1)
   
    return tax, nogating_info, nogating_rate, gating_info, gating_rate

def _plot_word_cond_info(tax, nogating_info, nogating_rate, gating_info, gating_rate):
    _plt.close('gating_vs_FEM_Word_cond_info')
    fig, ax = _plt.subplots(num='gating_vs_FEM_Word_cond_info')
    traces = []
    traces.append(ax.plot(tax, nogating_info, 'b', lw=2, label='no gating'))
    traces.append(ax.plot(tax, gating_info, 'r', lw=2, label='gating'))
    traces.append(ax.plot([0,0], ax.get_ylim(), 'k:', label='_nolabel_'))

    ax.legend(loc='center right', fontsize=10, bbox_to_anchor=(1,.71), frameon=False, handlelength=.5, handletextpad=.1)

    ax.set_xlim(-.1, .4)
    ax.set_xticks((-.1, 0, .1, .2, .3))
    ax.set_xticklabels((-.1,"", .1,"",.3), fontsize=10)
    ax.set_ylim(0, .4)
    ax.set_yticks((0, .25))
    ax.set_yticklabels((0, .25), fontsize=10)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlabel('Time (s)',fontsize=10)
    ax.set_ylabel('Information (Bits)',fontsize=10)
    ax.yaxis.set_label_coords(-.25, .40)
    fig.subplots_adjust(bottom=.3, left=.25, right=1, top=1)

    fig.set_size_inches(2, 1.5)
    fig.savefig('Figures/gating_vs_FEM_Word_cond_info.pdf', transparent=True)

    return fig, ax, traces

    _plt.close('word_rate')
    fig = _plt.figure('word_rate')
    ax = fig.add_subplot(1,1,1)
    ax.plot(nogating_tax, nogating_rate, 'b', lw=2, label='fem')
    ax.plot(gating_tax, gating_rate, 'r', lw=2, label='gating')
    ax.legend()

def _get_word_cond_info(letter_length, tax, gating_start_t, gating_end_t, g, covG, nogating_sig, gating_sig, binsN, gating_flag):
    '''
    I'm only implementing a 2 letter word.  At each point in time, compute I( last letter of response ; g | previous letter of response)

    '''
    info = _np.zeros(len(tax))
    rate = _np.zeros(len(tax))

    for i, t in enumerate(tax):
        prev_t = t - letter_length

        binned_g = _info.binned(g[:, time_to_point(t,0)], binsN)
        words = _get_words([prev_t, t], gating_start_t, gating_end_t, g, covG, nogating_sig, gating_sig, gating_flag, binsN=binsN)
        
        rate[i] = words.mean()
        info[i] = _info.cond_mi(words[:,1], binned_g, words[:,0])
    return info, rate

def _compare_gating_with_fem(t0, deltaT, g, covG, sig_FEM, sig_gating, binsN):
    '''
    I have the linear prediction g as a function of time.
    I'm going to compute FRs from g at t0 and t0-deltaT (two letters spaced by deltaT).
    In computing the FR I'm going to use at least two different sigmoids (one for gating, one for FEM)
    Then I'm going to compute the words with two letters and compute the information that the new letter carries about g given the previous letter.
    I'm going to construct three different words
        a. using both letters from no gatin sigmoid
        b. using both letters from gating sigmoid
        c. using one letter from gating, one from no gating
    '''
    # compute the binned firing rate at both t0 and t0-deltaT for each sigmoid in sigmoids
    letters = {}
    letters[('sig_FEM',0)]       = _info.binned(convert_to_firing_rate(t0, g, covG, [sig_FEM])[:,0], binsN)
    letters[('sig_FEM',-1)]      = _info.binned(convert_to_firing_rate(t0-deltaT, g, covG, [sig_FEM])[:,0], binsN)
    letters[('sig_gating', 0)]   = _info.binned(convert_to_firing_rate(t0, g, covG, [sig_gating])[:,0], binsN)
    letters[('sig_gating', -1)]  = _info.binned(convert_to_firing_rate(t0-deltaT, g, covG, [sig_gating])[:,0], binsN)

    
    # bin g
    p0 = time_to_point(t0, 0)
    binned_g = _info.binned(g[:, p0], binsN)
    '''
    _info.labels2prob(binned_g, output_flag=1)
    _info.labels2prob(letters[('sig_FEM', 0)], output_flag=1)
    _info.labels2prob(letters[('sig_FEM', -1)], output_flag=1)
    _info.labels2prob(letters[('sig_gating', 0)], output_flag=1)
    _info.labels2prob(letters[('sig_gating', -1)], output_flag=1)
    '''
    info = _np.zeros(3)
    fr = _np.zeros_like(info)

    info[0] = _info.cond_mi(letters[('sig_FEM',0)], binned_g, letters[('sig_FEM', -1)])
    info[1] = _info.cond_mi(letters[('sig_gating', 0)], binned_g, letters[('sig_gating', -1)])
    info[2] = _info.cond_mi(letters[('sig_gating', 0)], binned_g, letters[('sig_FEM', -1)])

    fr[0] = (letters[('sig_FEM', 0)].mean() + letters[('sig_FEM', -1)].mean())/2
    fr[1] = (letters[('sig_gating', 0)].mean() + letters[('sig_gating', -1)].mean())/2
    fr[2] = (letters[('sig_gating', 0)].mean() + letters[('sig_FEM', -1)].mean())/2
    return info, fr

def sigmoids_plot(name, tax, data, sigmoids):
    _plt.close(name)
    fig, ax = _plt.subplots(num=name)

    traces = []
    for i, sig in enumerate(sigmoids):
        traces.append(_plt.plot(tax, data[:,i], label=r'$T={0}, \sigma={1}$'.format(sig[2], sig[3])))

    _plt.legend()

    return fig, ax, traces

def explore_all_nl(g, time, nls):
    '''
    plot all nongating and gating nonlinearities simultaneously on top of the distribution of linear prediction values at the given times

    inptus:
    -------
        times:      list of times

        nls:        list of nonlinear_block objects
    '''
    #_pdb.set_trace()
    
    # close the plot if it already exists
    _plt.close('all_nls')
    fig, ax1 = _plt.subplots(num = 'all_nls')
    ax2 = ax1.twinx()

    colors = 'br'   # blue for no gating, red for gating
    bins=50
    point = time_to_point(time,0)
    data_to_hist = g[:,point]
    
    # normalize by mean and contrast
    data_to_hist -= data_to_hist.mean()
    data_to_hist /= data_to_hist.std()

    label = r'$t ={0: G}ms$'.format(int(1000*time))
    hist, bins, patches = ax1.hist(data_to_hist, bins=bins, normed=True, color='k', histtype='bar', label=label)

    #_pdb.set_trace()
    # plot nl in the same bins
    for nl in nls:
        ax2.plot(bins, nl.torate(bins)/nl.gating_nl.max_fr, colors[0], label=r'$no gating$', lw=1, alpha=.5)
        ax2.plot(bins, nl.gating_rate(bins)/nl.gating_nl.max_fr, colors[1], label=r'$gating$', lw=1, alpha=.5)
    
    #arange axis
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    ax1.set_xlabel('Filtered stimulus',fontsize=10)
    ax1.set_ylabel('',fontsize=10)
    ax1.yaxis.set_label_coords(-.25, .40)
    fig.subplots_adjust(bottom=.25, left=.10, right=.9, top=.95)

    #fig.set_size_inches(2, 1.5)
    fig.savefig('Figures/gating_vs_FEM_Word_cond_info.pdf', transparent=True)

    ax1.set_xticks((-1, 0, 1))
    ax1.set_xticklabels((-1, 0, 1), fontsize=10)
    #ax.set_xlim(-1,1)
    ax1.set_yticks((0, ax1.get_ylim()[1]))
    ax1.set_yticklabels((0,ax1.get_ylim()[1]), fontsize=10)

    ax1.text(0.2, 4, r'$t=-100ms$',color='b')
    ax1.text(0.2, 3, r'$t= 100ms$', color='r')

    ax2.set_ylim(0,ax2.get_ylim()[1])
    #ax.legend(bbox_to_anchor=(1.4,.75), fontsize=10, handlelength=.5, frameon=False)
    fig.savefig('Figures/LP_and_sigmoids.pdf', transparent=True)
    
    return fig, ax1

def reproduce_Yusuf(plot_flag=1):
    '''
    From the experiment in Igor, I have exported a text file with 4 fields per cell:
        mp: stands for membrane potential
        stim_sd:    standard deviation of the photodiode
        mp_avg:     membrane potential average
        avg_mp_sd:  sd across time of the average membrane potential. It has nothing to do with noise
        mp_noise:   mp sd across repeats of the same stimulus, is the noise of the membrane potential
    
    output:
    -------
        sim_noise_fit:      A linear fit to sim_noise_sd vs sim_mp_sd
                            Both sim_mp_sd and sim_noise_sd are in the same units as the simulation
                            Given a set of linear prediction, compute the SD and the noise to use is sim_noise_fit( linear_prediction.std() )
    '''
    #_pdb.set_trace()
    bipolar_cell = cell()

    # load parameters from text file
    df = _pd.read_csv('bipolar_cell.txt', sep=' ').sort('stim_sd').reset_index(drop=True)#, index_col='stim_sd')
    
    # Reproduce the data with the simulation
    for i, contrast in enumerate(df['stim_sd']):
        # compute the sd of the simulated membrane potential when using a gaussian distribution signal of the same contrast as the one used in the real experiment
        df.set_value(i, 'sim_mp_sd', bipolar_cell.filter_gaussian_contrast(contrast).std())

        # now scale the noise such that the ratio between noise/mp_sd is the same in the experiment and in the simulation
        df.set_value(i, 'sim_mp_noise', df.get_value(i, 'sim_mp_sd')*df.get_value(i, 'exp_mp_noise')/df.get_value(i, 'exp_mp_sd'))

    # fit a lines between all the values of sim_mp_noise and sim_mp_sd
    sim_noise_fit = _np.poly1d(_np.polyfit(df['sim_mp_sd'], df['sim_mp_noise'],1))

    # the connection between the experiment and the simulation are the gaussian contrasts. For each gaussian contrast, express the noise as a fraction 
    if plot_flag:
        _plt.close('noise SD to stim SD')
        fig, ax = _plt.subplots(num='nose SD to stim SD')
        ax.plot(df['sim_mp_sd'], df['sim_mp_noise'], 'ok')
        ax.plot([10,120], [sim_noise_fit(10), sim_noise_fit(120)], 'k')
        ax.set_xlabel('simulted stimulus sd')
        ax.set_ylabel('simulated noise sd')

    return sim_noise_fit, df
    
def explore_one_cell_nl(g, time, nls, bin_rate=None):
    '''
    plot the distribution of values of g at time t0 and all nonlinear objects in nls

    nls:         iterable of nonlinear_block objects

    bin_rate:   if given, rate is discretized

    '''
    _pdb.set_trace()
    # close the plot if it already exists
    _plt.close('LP_and_sigmoid')
    fig, ax1 = _plt.subplots(num = 'LP_and_sigmoid')
    ax2 = ax1.twinx()

    colors = 'rb'   # blue for no gating, red for gating
    bins=500
    point = time_to_point(time,0)
    data_to_hist = g[:,point].T
    
    # normalize by mean and contrast
    #data_to_hist -= data_to_hist.mean()
    #data_to_hist /= data_to_hist.std()
    
    label = r'$t ={0: G}ms$'.format(int(1000*time))
    #hist, bins, patches = ax1.hist(data_to_hist, bins=bins, normed=True, histtype='step', color='k', histtype='bar', label=label)
    hist, bins, patches = ax1.hist(data_to_hist, bins=bins, histtype='step', normed=True, color='k', label=label)

    # plot nl in the same range
    bins = _np.arange(bins[0], bins[-1], (bins[-1]-bins[0])/1000)
    if not _np.iterable(nls):
        nls = [nls]

    for i, nl in enumerate(nls):
        ax2.plot(bins, nl.torate(bins, bin_rate=bin_rate), colors[0], lw=2)

    #arange axis
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    ax1.set_xlabel('Filtered stimulus',fontsize=10)
    ax1.set_ylabel('',fontsize=10)
    ax1.yaxis.set_label_coords(-.25, .40)
    fig.subplots_adjust(bottom=.25, left=.10, right=.9, top=.95)

    #fig.set_size_inches(2, 1.5)
    fig.savefig('Figures/gating_vs_FEM_Word_cond_info.pdf', transparent=True)

    #ax1.set_xticks((-4, -2, 0, 2, 4))
    #ax1.set_xticklabels((-4, "", 0, "", 4), fontsize=10)
    #ax.set_xlim(-1,1)
    ymax = ax1.get_ylim()[1]
    ax1.set_yticks((0, ymax))
    ax1.set_yticklabels((0,ymax), fontsize=10)

    ax1.text(0.2, 4, r'$t=-100ms$',color='b')
    ax1.text(0.2, 3, r'$t= 100ms$', color='r')

    ax2.set_ylim(0, ax2.get_ylim()[1])
    #ax.legend(bbox_to_anchor=(1.4,.75), fontsize=10, handlelength=.5, frameon=False)
    fig.savefig('Figures/LP_and_sigmoids.pdf', transparent=True)
    
    return fig, ax1

def info_between_fr_and_g(firing_rate, g, t1, binsN=8):
    '''
    Given the firing rate at a given time and the linear prediction (g), compute the information between the firing_rate and the linear prediction at time t1


    Compute cor(g(t>t0); firing_rate(t0))

    inputs:
    -------
        firing_rate (ndarray):  Firing rate at time t0, probably the output of convert_to_firing_rate

        g (2D ndarray):         linear prediction, simulation's output
        
        t1 (float):             correlation between firing rate and g will be computed for all times in between t0 and t1 (inclusive)
    '''

    # convert t1 to point in g
    p1 = time_to_point(t1, 0)
    
    #_pdb.set_trace()
    # bin firing rate and g using binsN
    binnedFR = _info.binned(firing_rate, binsN)
    binned_g = _info.binned(g[:,p1], binsN)

    return _info.mi(binnedFR, binned_g)
    #cov = _np.cov(firing_rate, g[:, p1])
    #return _info.gaussianInformation(cov, [0], [1])

def info_decay(t0, t1, g, covG, sigmoids, binsN=8):
    '''
    For the given parameters, compute the firing rate (at time t0) for the given thresholds and sigmas.
    For each Firing rate then estimate the information conveyed about 'g' at latter times (up to t1)

    inputs:
    -------
        t0 (float):         time at which firing rate is computed

        t1 (float):         last time at which information between g and firing rate is computed

        g (2D ndarray):     linear prediction

        sigmoids (list):    list of tuples with (threshold, sigma) for each sigmoid I want to use
    '''
    # make a plot with histogram of linear prediction and sigmoids
    explore_sigmoid(g, t0, sigmoids)

    # get the x axis for the plot
    tax = list(_np.arange(t0, t1, sim_delta_t))

    # preallocate ndarray for decay of info
    decay = _np.zeros((len(tax), len(sigmoids)))
    
    # compute the firing rate at time t0 for the given sigmoid
    #firing_rate = convert_to_firing_rate(sig[0], sig[1], g, t0)
    firing_rate = convert_to_firing_rate(t0, g, covG, sigmoids)

    plot_firing_rate_hist(firing_rate)

    # for each time in between t0 and t1, compute the information
    for i in range(len(sigmoids)):
        print('processing sigmoid {0}, {1} out of {2}'.format(sigmoids[i], i+1, len(sigmoids)))
        for t in tax:
            decay[(t-t0)/sim_delta_t, i] = info_between_fr_and_g(firing_rate[:, i], g, t, binsN=binsN)

    sigmoids_plot('decay', tax, decay, sigmoids)
    _plt.title('binsN={0}'.format(binsN))
    _plt.savefig('Figures/Decay, binsN={0}.pdf'.format(binsN), transparent=True)
    return decay

def info_gained_by_last_letter(covG, covN, letter_length, lettersN):
    '''
    Return the informatin that the last letter (out of lettersN) conveys about g given all previous letters

    The information I'm computing is:
        I(g(p0) ; g(p0)+n | g(p1)+n, ..., g(pn)      where p1, p2, ..., pn are points distanced by t0 seconds

    Implemenation notes:
        I(X; Y | Z) = H(X | Z) - H(X | Y, Z)
                    = H(X, Z) - H(Z) - ( H(X, Y, Z) - H(Y, Z) )
                    = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
     computes:
        I(g(t), g(t)+noise)
        I(g(t), g(t)+noise | g(t-t0)+noise)                                   
        I(g(t), g(t)+noise | g(t-t0)+noise,g(t-2*t0)+noise)                   
        I(g(t), g(t)+noise | g(t-t0)_noise,g(t-2*t0)+noise,g(t-3*t0)+noise)   
    
    intpus:
    -------
        covG (2d ndarray)

        letter_length (float):  in seconds

        lettersN (int):         computes the information between g(t0) and a word of length lettersN taken from g(t)+noise

        save (string):          if given will save the output information to the given file
                                if not given, saves chain information to a default name: 'chainInfo_{0}L_{1}ms'.format(lettersN, int(t0*1000)),
    
    Returns:
    --------

    '''
    #_pdb.set_trace()
    # pre allocate ndarrays for output
    tax = _getTAX()
    cond_info = _np.zeros_like(tax)

    for i, t in enumerate(tax):
        if lettersN == 1:
            # in this case there is no cond_info
            X = _extractSubCov(covG, covN, [], [i])
            Y = _extractSubCov(covG, covN, [i], [])
            XY = _extractSubCov(covG, covN, [i], [i])

            entropy_X   = _info.gaussianEntropy(X)
            entropy_Y   = _info.gaussianEntropy(Y)
            entropy_XY  = _info.gaussianEntropy(XY)

            cond_info[i] = entropy_X + entropy_Y - entropy_XY
        else:

            # Define the points (and their time counter parts) that make the word
            times = [t - i*letter_length for i in range(lettersN-1, -1, -1)]
            if times[0] < tax[0]:
                cond_info[i] = _np.nan
            else:
                points = [time_to_point(t, 0) for t in times]

                # in order to compute information, I need to extract the covariance matrix between g(t) and g+n(t), g+n(t-1), etc.
                # then mi_1 is a conditional information and mi_2 is not
                # According to the implementaion note, I will need to compute 4 different entropies in the conditional information. Extract the subCov corresponding for each one of them
                # X is the noisy letter at t
                # Y is the true value of the linear prediction at t
                # Z are the noisy letters at previous times than t
                XZ = _extractSubCov(covG, covN, [], points)
                YZ = _extractSubCov(covG, covN, [points[-1]], points[:-1])
                XYZ = _extractSubCov(covG, covN, [points[-1]], points)
                Z = _extractSubCov(covG, covN, [], points[:-1])

                entropy_XZ  = _info.gaussianEntropy(XZ)
                entropy_YZ  = _info.gaussianEntropy(YZ)
                entropy_XYZ = _info.gaussianEntropy(XYZ)
                entropy_Z   = _info.gaussianEntropy(Z)

                cond_info[i] = entropy_XZ + entropy_YZ - entropy_XYZ - entropy_Z
            
    cond_info.tofile('Data/cond_info_{0}L_{1}ms'.format(lettersN, int(1000*letter_length)), sep="\r", format="%5f")

    return cond_info

def plot_compare_letters_N(letterLength, lettersN_list=None, nameout = None):
    '''
    Figure to compare I(g(t0); g(t0)+n | all other letters in the word)
    
    Search in current directory for files of the form 'cond_info_{0}L_{1}ms where {0} is the number of letters and {1} is the length of each letter

    files 'chainInfo_{0}L_{1}ms were processed with info_gained_by_last_letter
    output:
    -------
        Generates and saves plot 'Figures/LettersN'
        
    '''
    from os import listdir
    # generate the plot, killing it if it already exists
    if nameout is None:
        nameout = 'compare_letters_N'

    _plt.close(nameout)
    fig, ax = _plt.subplots(num=nameout)

    #_pdb.set_trace()
    # get all the files in current folder with MI of word and linear prediction decompossed as contributions of each letter
    cond_files = [name for name in listdir('Data/') if name.startswith('cond_info') and name.endswith('_{0}ms'.format(letterLength))]
    cond_files.sort(key=lambda x: int(x.split('_')[2][:-1]))       # in this line [2] is for token out of split and [:-1] is to remove last L in 2nd token

    # get the tax if it doesn't exist
    tax = _getTAX()
    traces = []
    #labels = []
    for name in cond_files:
        info = _np.fromfile('Data/'+name, sep='\r').reshape(300,-1)
        tokens = name.split('_')
        lettersN = tokens[2]
        
        # lettersN is a number + "L" at the end.
        # if letterN_list was given, I want to know if current lettersN from file is in letterN_list. For that I need to convert to int lettersN after removing the last char ("L")
        if lettersN_list is None or int(lettersN[:-1]) in lettersN_list:
            #labels.append('L={0}'.format(lettersN))
            traces.append(ax.plot(tax, info[:,-1], linewidth=2, label='L={0}'.format(lettersN)))

    
    # add saccade dotted line
    ax.plot((0,0), (0, 2), 'k:', label='_nolegend_')

    ax.set_xlim(-.2, .8)
    ax.set_ylim(-.1, ax.get_ylim()[1])
    #ax.legend(loc=9, fontsize=10, ncol=3, handlelength=1, handletextpad=.25, columnspacing=1, frameon=False, bbox_to_anchor=(.5,1.10))#, frameon=False)
    ax.legend(loc='center right', fontsize=10, handlelength=1, handletextpad=.25, frameon=False, bbox_to_anchor=(1,.7))#, frameon=False)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Information (Bits)', fontsize=10)
    ax.yaxis.set_label_coords(-.15, .4)
    ax.set_title('g( g(t)+n; g(t) | all other letters)\ncomparing different word lengths')
   
    ax.set_xticks((0, .5))
    ax.set_xticklabels((0, .5), size=10)
    ax.set_yticks((0, 1))
    ax.set_yticklabels((0,1), size=10)

    
    # add margin for axis labels
    fig.subplots_adjust(bottom=.3, left=.2, right=1, top=1)
    fig.set_size_inches(2.5, 1.5)
    fig.savefig('Figures/{0}.pdf'.format(nameout))

    return fig, ax

def plot_compare_letter_length(lettersN, length_list=None):
    '''
    Figure to compare I(g(t0);g(t0)+n | all ohter letters) for words of a fixed number of letters, where the letterLength changes

    output:
    -------
        generates and saves plot 'Figures/LettersN'
    '''

    from os import listdir
    
    _plt.close('letterLength')
    fig, ax = _plt.subplots(num='letterLength')

    cond_files = [name for name in listdir('Data/') if name.startswith('cond_info_{0}L_'.format(lettersN))]
    cond_files.sort(key=lambda x: int(x.split("_")[3].strip('ms')))

    tax = _getTAX()
    traces = []

    #_pdb.set_trace()

    for name in cond_files:
        info = _np.fromfile('Data/'+name, sep='\r').reshape(300,-1)
        tokens = name.split('_')

        # I need to compare tokens[-1] (the one ending in 'ms') with the ones in lenght_list (if not None). 
        if length_list is None or int(tokens[-1].rstrip('ms')) in length_list:
            traces.append(_plt.plot(tax, info[:]/info[40:260].max(), linewidth=2, label='{0}'.format(tokens[3])))

    traces.append(_plt.plot((0,0), (0, ax.get_ylim()[1]), 'k:', label='_nolabel_'))

    ax.set_xlim(-.2, .8)
    ax.set_ylim(0, 1.2)
    ax.set_xticks((0, .5))
    ax.set_xticklabels((0, .5), size=10)
    ax.set_yticks((0, 1))
    ax.set_yticklabels((0, 1), size=10)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.legend(fontsize=10, handlelength=1, handletextpad=.25, frameon=False, bbox_to_anchor=(1,1.1))#, frameon=False)
    #ax.legend(loc=9, fontsize=10, ncol=3, handlelength=1, handletextpad=.25, columnspacing=1, frameon=False, bbox_to_anchor=(.5,1.10))#, frameon=False)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Normalized\nInformation', fontsize=10)
    fig.subplots_adjust(bottom=.30, left=.25, right=1, top=1)
    ax.set_title('I( g(t)+n; g(t) | all other letters and lettersN={0})\ncomparing different letters length'.format(lettersN))
    
    fig.set_size_inches(2.5, 1.5)
    fig.savefig('Figures/letterLengthRedundancy.pdf')

    return fig, ax

def fit_exp_to_simulation(g, df, nogating_t, gating_t, cell=None):
    '''
    Load the sigmoidal nonlinearities form the experiments (during gating and FEM) and change their scaling to match the simulation
    
    inputs:
    -------
        g:                  the linear predictions from the simulation

        df (pandas df):     a data frame with all nonlinearities for all cells. Most likely the output of "load_sigmoids(length=100)"

        nogating_t (float):      time at which to fit experimental nogating_sig

        gating_t (float):   time at which to fit expeirmental gating_sig
    
        cell (int):         which cell's NL to load, has to be in the range of the df
                            if not given a random one is picked

    outputs:
    --------
        nogating_sig:            parameters for the sigmoidal nonlinearity

        gating_sig:         idem nogating_sig


    Experimental sigmoids were calculated in igor and are in units of contrast (1 in the x axis means 1 standard deviation)
    sigmoids are given with 7 numbers and they represent:
    sig[0] + sig[1]/(1+exp(-(x-sig[2])/sig[3]))
    sig[4]:     leftx of nonlinearity
    sig[5]:     rightx of nonlinearity
    sig[6]:     experimental contrast used
    '''

    """
    # Load sigmoids from 3%
    exp_nogating_sig = _np.fromfile('d100928_R1_c19_0_NL_0Y.txt', sep='\r')
    exp_gating_sig = _np.fromfile('d100928_R1_c19_0_NL_1Y.txt', sep='\r')
    """

    #_pdb.set_trace()

    if cell is None:
        cell = _np.random.randint(len(df.index))
        print('fitting sigmoids for cell in df.iloc = {0}'.format(cell))

    # even though I'm only going to use in this script threshold and sd of sigmoid (points 2 and 3) I get out of the sigmoid all 4 points to avoid confusions later on when calling w[2] and w{3]
    exp_nogating_sig = df.iloc[cell][['TW0_w[0]', 'TW0_w[1]', 'TW0_w[2]', 'TW0_w[3]']].values

    if df['gatingTW'][cell] == 1:
        exp_gating_sig = df.iloc[cell][['TW1_w[0]', 'TW1_w[1]', 'TW1_w[2]', 'TW1_w[3]']].values
    elif df['gatingTW'][cell] == 2:
        exp_gating_sig = df.iloc[cell][['TW2_w[0]', 'TW2_w[1]', 'TW2_w[2]', 'TW2_w[3]']].values
    elif df['gatingTW'][cell] == 3:
        exp_gating_sig = df.iloc[cell][['TW3_w[0]', 'TW3_w[1]', 'TW3_w[2]', 'TW3_w[3]']].values

    # find out the SD of the linear prediction at the given times
    nogating_SD = g[:, time_to_point(nogating_t,0)].std()
    gating_SD = g[:, time_to_point(gating_t, 0)].std()

    nogating_sig = exp_nogating_sig
    gating_sig = exp_gating_sig

    # if sigmoid's std is zero, just let it be, don't divide by 0 creating nans
    if exp_nogating_sig[3] != 0:
        nogating_sig[2:4] *= nogating_SD/exp_nogating_sig[3]
    
    if exp_gating_sig[3] != 0:
        gating_sig[2:4] *= gating_SD/exp_gating_sig[3]

    """
    print('printing FEM results')
    print(exp_nogating_sig)
    print(nogating_SD)
    print(nogating_sig)
    print('\r')
    print('printing Gating results')
    print(exp_gating_sig)
    print(gating_SD)
    print(gating_sig)
    """
    return nogating_sig, gating_sig


def test_several_sigmoids(g, covG, df, n=None):
    '''
    compute "get_word_cond_info" for "n" randomly choosen cells from all cells in df (a pandas data frame). If n is none it just computes it across all cells in df
    '''
    from time import time

    #_pdb.set_trace()

    if n is None:
        n_list = range(len(df.index))
    else:
        n_list = _np.random.randint(0, len(df.index), n)

    cond_info = []
    for i, n in enumerate(n_list):
        t0 = time()
        print("processing cell {0}".format(n))
        cond_info.append(get_word_cond_info(g, covG, fit_exp_to_simulation(g, df, -.1, .1, cell=n)))
        print("{0} took {1} secs to run".format(n, time()-t0))

    return cond_info

def _fix_cond_info(cond_info):
    '''
    change cond info from list of tuples of ndarrays to be ndarray.
    Then remove all the many identical tax keeping just one.
    Perform stats on fem and gating

    inputs:
    -------
        cond_info:      output of test_several_sigmoids
    '''
    tax = cond_info[0][0]

    fem = _np.zeros((len(cond_info), cond_info[0][0].shape[0]))
    gating = _np.zeros((len(cond_info), cond_info[0][0].shape[0]))

    for i, tup in enumerate(cond_info):
        fem[i,:] = tup[1]
        gating[i,:] = tup[3]

    nogating_mean = fem.mean(axis=0)
    nogating_std = fem.std(axis=0)
    gating_mean = gating.mean(axis=0)
    gating_std = gating.std(axis=0)

    fem = (fem, nogating_mean, nogating_std)
    gating = (gating, gating_mean, gating_std)
    
    return tax, fem , gating

def fake_gaussian_noise(contrast, samples=1E5, mean=127):
    '''
    Fake a stimulus similar to the one used in UFlicker experiments
    

    output:
    -------
        stim (1D ndarray):        sequence of light intensities
    '''
    #_pdb.set_trace()

    if contrast>=1:
        contrast/=100

    # grab random number with 0 mean and STD=1
    stim = _np.random.randn(samples)*mean*contrast + mean

    # arange stim so that each intensity value lasts ~30ms. The simulation is set up such that each frame lasts sim_delta_t
    stim = stim.reshape(-1, 1) * _np.ones((1, int(.03/sim_delta_t)))
    stim = stim.reshape(-1, )
    
    return stim
   

"""
def _fit_contrast_to_simulation(g, time):
    '''
    Find the gaussian contrast that best approximates the distribution of linear predictions (g) at 'time'

    *********************************
        This is not working as expected because the only thing this is doing is trying to remove the huge pick at 0 in the gaussian distribution by increasing the contrast
    *********************************

    outputs:
    --------
        return the contrast Gaussian fit
    '''

    point = time_to_point(time, 0)
    hist, bins = _np.histogram(g[:,point], bins=200)
    errors = []
    
    for C in _np.arange(7,10,.5):
        gaussian = filter_gaussian_noise(filter_instance, C)
        hist_g, _ = _np.histogram(gaussian, bins=bins)

        hist_g -= hist
        errors.append((C,_np.dot(hist_g, hist_g))

    return errors

def _threshold_to_rate(g, threshold, slope=1):
    '''
    convert linear prediction to rate using just rectification. the nonlinearity is described by a threshold. For a value of g rate is 0 if g < threshold and (g-threshold)*slope otherwise

    intpus:
        g (ndarray):        can have any number of dimensions.

        threshold (float):  

        slope (float):

    output:
        rate (ndarray):     same dimension as g

    '''
    thresholded_g = (g - threshold)*slope
    below_thresh_indices = thresholded_g< 0
    thresholded_g[below_thresh_indices] = 0
    
    return thresholded_g

def _threshold_to_rate2(g, threshold, slope=1):
    return _np.where(g>threshold, (g-threshold)*slope, _np.zeros_like(g))
"""

def _get_information_ratio(sub_g, nogating_threshold, gating_threshold, binned_g, binsN, lettersN=2):
    '''
    Get the ratio between two informations, gating over no-gating
    Independently of gating or no-gating, each word has the same number of letters (2 by default).
    Timing of both words is the same and is such that all letters but the last one are before gating and the last one is after gating
    1st word is such that all letters are computed using non gating nonlinearity (denominator)
    2nd word is computed with just the last letter under the gating nonlinearity. Gating rates are computed by shifting the threshold from the non gating nonlinearity by 10x the STD of the linear prediction at 3% contrast (that comes from my model done in igor)
    
    inputs:
    -------
        sub_g:  already selected for the two times of interest, it's shape has to be (:,2), oterwise raise error

    by looping over this function with many different thresholds, I can get how efficient gating is. Pass the output of this into a conditional infomration function
    '''

    #_pdb.set_trace()
    # convert g to firing rate in the case of no gating
    nogating_words = _threshold_to_rate(sub_g, nogating_threshold)

    # convert g to firing rate in the case of gating
    gating_words = nogating_words.copy()
    gating_words[:,1] = _threshold_to_rate(sub_g[:,1], gating_threshold)

    # bin both type of words using the same max/min
    v_max = max(gating_words.max(), nogating_words.max())
    v_min = min(gating_words.min(), nogating_words.min())

    nogating_words_binned = _info.binned(nogating_words, binsN, maxX=v_max, minX=v_min)
    gating_words_binned = _info.binned(gating_words, binsN, maxX=v_max, minX=v_min)

    return _info.cond_mi(binned_g, gating_words_binned[:, -1], gating_words_binned[:, 0])/_info.cond_mi(binned_g, nogating_words_binned[:, -1], nogating_words_binned[:,0])

    
    cond_info = []
    for lettersN in letter_number_list:
        pass

    return cond_info

def get_information_ratio(g, last_letter_t, letter_length, binsN, lettersN=2):
    '''
    wrapper to call _get_information_ratio
    '''
    #_pdb.set_trace()

    # convert times of letters to points
    points = [time_to_point(last_letter_t - n*letter_length, 0) for n in range(lettersN-1, -1, -1)]

    # extract teh values of g at those points of interest
    sub_g = g[:, points]

    # bin the linear prediction at last_letter_t
    binned_g = _info.binned(sub_g[:, -1], binsN)

    # calculate how much is the nonlinearity shifting during gating
    shift = _get_shift()
    no_gating_thresh = 50
    gating_thresh = no_gating_thresh - shift

    return _get_information_ratio(sub_g, no_gating_thresh, gating_thresh, binned_g, binsN, lettersN = lettersN)
    
def rate_increase(g, slope_range, thresh_range, binsN):
    '''
    compute the ratio of gating to nongating information for a bunch of conditions
    '''
    rate_increase_array = _np.zeros((len(slope_range), len(thresh_range)))

    for i, slope in enumerate(slope_range):
        for j, thresh in enumerate(thresh_range):
            nl = nonlinear_block('sigmoid', {'thresh':thresh, 'slope':slope})

            _, _, rate_increase_array[i, j] = compute_gating_effect(g, [.08, .1], nl, binsN)

    return rate_increase_array

def plot_rate_increase(rate_array):
    """_plt.close('rate_increase')
    fig, ax = _plt.subplots(num='rate_increase')
    ax.matshow(rate_array)

    cbar = fig.colorbar(ax, interpolation='nearest', vmin=0.5, vmax=.99)#, ticks=[1, 1, 2])
    #cbar.ax.set_yticklabels(['< -1', '0', '> 1'])# vertically oriented colorbar
    """
    _plt.imshow(rate_array)
    _plt.colorbar()

def compute_gating_effect(g, letter_times, nl, binsN):
    '''
    * bin g using binsN
    # get words by passing g at letter_times through nonlinearity. Last letter gets also passed through gating_nl
    * bin words using binsN
    * compute Shannon's I(last letter; g | previous letter)
    '''

    #_pdb.set_trace()
    time_points = [time_to_point(t, 0) for t in letter_times]
    
    # try binning g in an intelligent way, using percentiles. Each bin takes 100/2**binsN chuncks of data
    bins = list(_np.arange(0, 100.1, 100/binsN)) 
    percentiles = _np.percentile(g[:, time_points[-1]], bins)
    g_binned = _np.digitize(g[:, time_points[-1]], percentiles)

    # pass linear prediction at the letters of interest through nonlinearity
    bin_rate = g.max()/binsN
    words = nl.torate(g[:, time_points], bin_rate = bin_rate)
    gating_letter = nl.gating_rate(g[:, time_points[-1]], bin_rate = bin_rate)
    
    if _np.isnan(max(words.flatten())) or _np.isnan(max(gating_letter)):
        raise ValueError('eihter words or gating_letter got "NaN"s inside compute_gating_effect')

    non_gating_info =  _info.cond_mi(words[:, -1], g_binned, words[:,0])
    gating_info = _info.cond_mi(gating_letter, g_binned, words[:,0])
    return gating_info, non_gating_info

def wrap_gating_effect(g, letter_length, nl, binsN):
    '''
    wrapper to call compute gating_effect with 2L words at all possible times

    outptu:
    -------
        save to file "Data/gating_effect_2L_{letter_length}ms" and "Data/nogating_effect_2L_{letter_length}ms"
    '''
    tax = _getTAX()

    # allocate output
    gating = _np.zeros_like(tax)
    nogating = gating.copy()

    start_p = letter_length/sim_delta_t
    for p, t in enumerate(tax):
        if p < start_p:
            continue

        letter_times = [t-letter_length, t]
        gating[p], nogating[p] = compute_gating_effect(g, letter_times, nl, binsN)
        
    gating.tofile('Data/gating_effect_2L_{0}ms'.format(int(1000*letter_length)))
    nogating.tofile('Data/nogating_effect_2L_{0}ms'.format(int(1000*letter_length)))

def plot_gating_effect(letter_length):
    from os import listdir

    # get the list of files in 'Data/' with gating_effect_2L_{0}ms
    gating_file = [name for name in listdir('Data/') if name.startswith('gating_effect_2L_{0}ms'.format(int(1000*letter_length)))][0]
    nogating_file = [name for name in listdir('Data/') if name.startswith('nogating_effect_2L_{0}'.format(int(1000*letter_length)))][0]

    gating = _np.fromfile('Data/'+gating_file)
    nogating = _np.fromfile('Data/'+nogating_file)

    tax = _getTAX()
    _plt.close('gating_effect')
    fig, ax = _plt.subplots(num='gating_effect')

    ax.plot(tax, gating, 'b', lw=2)
    ax.plot(tax, nogating, 'r', lw=2)

    ax.set_xlim(-.2, .8)

    fig.set_size_inches(2, 2)
    fig.savefig('Figures/gating_effect.pdf')


def load_model_fit():
    '''
    In natural_scenes_fitting I loaded all PSTHs corresponding to a cell (all contrasts) and fitted a model where the threshold and peripheral_weight were variables.

    All results were stored in file: 'UFlicker PSTHs/best_parameters.txt' which is composed of 4 fields, the cell #, peripheral_weight, nl_threshold, and the error between the fit and teh PSTH
    
    Load that information and create and return a dictionary with cell # as key and a tuple with nonlinearity objects as values. Nonlinearities are of 'birect' form with just a threshold. Base nonlinearity uses nl_thresh as threshold and gated_nl has nl_thresh + peri_weight.

    dict[0] = (base_nonlinearity, gated_nonlinearity)
    '''
    df = _pd.read_csv('UFlicker PSTHs/best_parameters.txt', sep=' ')
    
    #_pdb.set_trace()
    nls = {}
    for i in df.index:
        base_nl = nonlinear_block('birect', df.iloc[i]['nl_thresh'], nl_units)
        gated_nl = nonlinear_block('birect', df.iloc[i]['nl_thresh']-df.iloc[i]['peri_weight'], nl_units)

        nls[df.iloc[i]['cell_id']] = (base_nl, gated_nl)

    return nls

def nls_to_list(nls, sele = None):
    '''
    After loading all model fits into nls, nls is a dictionary with cells as keys and tuples as values. Each tuple holds 2 nonlinearity objects. The first for basal condition and the second for gating.

    I want to get a list with either all the nonlinearities, or those from 'basal' or 'gating'

    '''

    #_pdb.set_trace()
    nl_list = []
    for nl in nls.values():
        if sele is None or sele is 'basal':
            nl_list.append(nl[0])
        if sele is None or sele is 'gating':
            nl_list.append(nl[1])

    return nl_list


def load_sigmoids(s_file = 'UFlicker_sigmoids.txt', length=100, contrast=3):
    '''
    Load all sigmoidal fits to gating cells from UFlicker experiment and convert them to nonlinear_block objects.
    '''

    dataframe = _load_sigmoids_dataframe(s_file=s_file, length=length, contrast=contrast)
    return _dataframe_to_nonlinear_block_list(dataframe)

def _load_sigmoids_dataframe(s_file = 'UFlicker_sigmoids.txt', length=None, contrast=3):
    '''
    Load all sigmoidal fits to gating cells from UFLicker experiment

    inputs:
    -------
        s_file:         plain txt file exported from igor with
                        day retina length cell contrast mask TW0_w[0] TW0_w[1] TW0_w[2] TW0_w[3] TW1_w[0] TW1_w[1] TW1_w[2] TW1_w[3] TW2_w[0] TW2_w[1] TW2_w[2] TW2_w[3] TW3_w[0] TW3_w[1] TW3_w[2] TW3_w[3]
        
        length:         length in seconds of the UFlicker experiment to use, I usually work only with 100s

        contrast:       which contrast to load

    output:
    -------
        newDF (dataframe):  Dataframe with parameters for all experimental nonlinearities
                            Either work directly with it or call _dataframe_to_nonlinear_block_list(newDF)
    '''

    # first load all sigmoids
    df = _pd.read_csv(s_file, sep=' ', parse_dates=['day'])

    # restrict df to those sigmas with contrast == 3 and mask==1 and no no null sd
    #df_3 = df[(df['contrast']==contrast) & (df['mask']==1) & (df['TW0_w[3]']!=0) & (df['TW1_w[3]']!=0) ]
    df_3 = df[(df['contrast']==contrast) ]

    if length is not None:
        df_3 = df_3[df_3['length']==length]

    # I have manually selected a bunch of gating cells to run the analysis on.
    # I will only keep those
    # not nice but effective
    newDF = _pd.DataFrame()
    for cell in [1,2,4,10,11,12,15,16,17,18,19,20,21]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='100928') & (df_3['retina']==1) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)

        if cell == 1:
            newDF['gatingTW'] = _pd.Series(1, index=newDF.index)
        # add a column with information on which TW to use for gating
        if cell == 1:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 2)
        else:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element

    for cell in [1,2,6,7,9,21,23,24]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='101011') & (df_3['retina']==2) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)
        # add a column with information on which TW to use for gating
        if cell == 1:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 2)     # change only last element
        elif cell == 6:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 3)     # change only last element
        else:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element

    for cell in [1,6,7,12,14,15,18,21]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='101206') & (df_3['retina']==1) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)
        # add a column with information on which TW to use for gating
        if cell == 1:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 3)     # change only last element
        elif cell in [15,21]:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element
        else:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 2)     # change only last element
    
    for cell in [1,2,3,6]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='110106') & (df_3['retina']==2) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)
        # add a column with information on which TW to use for gating
        if cell == 2:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 2)     # change only last element
        else:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element
        
    for cell in [1,5,8,9,35]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='110204') & (df_3['retina']==1) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)
        newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element
    
    for cell in [1,4,7,10,17,18,20,22]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='110420') & (df_3['retina']==2) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)
        newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element

    for cell in [1,2,3,4,6,7,8]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='110516') & (df_3['retina']==1) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)
        newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element

    #newDF.append((df_3['day']==100928) & (df_3['retina']==1) & (df_3['cell'] in [2,4,10,11,12,15,16,17,18,19,20,21]))
    
    newDF.reset_index()

    return newDF

def _dataframe_to_nonlinear_block_list(df):
    '''
    convert data frame with all nonlinearities to a list of nonlinear_block objects
    '''
    sigmoids = []

    #_pdb.set_trace()
    # loop through the data frame and for each line, create a nonlinear_block object that has TW0 as the base nonlinearity and the one pointed at by gatingTW as the gating one.
    for i in df.index:
        row = df.iloc[i]

        # original nonlinearities are in contrast units (contrast in the 0-1 range). Remove that, dividing both threshold and sd by the contrast
        sigmoid = nonlinear_block('sigmoid', row['TW0_w[2]'], 'linear prediction', contrast = row['contrast'], min_fr = row['TW0_w[0]'], max_fr = row['TW0_w[1]']+row['TW0_w[0]'], sd=row['TW0_w[3]'])
        sigmoid.gating_nl = nonlinear_block('sigmoid', row['TW{0}_w[2]'.format(int(row['gatingTW']))], 'linear prediction', contrast = row['contrast'], 
                min_fr = row['TW{0}_w[0]'.format(int(row['gatingTW']))], 
                max_fr = row['TW{0}_w[1]'.format(int(row['gatingTW']))] + row['TW{0}_w[0]'.format(int(row['gatingTW']))], 
                sd = row['TW{0}_w[3]'.format(int(row['gatingTW']))])

        sigmoids.append(sigmoid)

    return sigmoids

def explore_word_letters(g, start_t, end_t, nl, bin_rate):
    '''
    plot a histogram of letters during the start_t, end_t under both gating and no gating nonlinearities
    '''
    #_pdb.set_trace()
    start_p = time_to_point(start_t, 0)
    end_p = time_to_point(end_t, 0)

    # limit g to the gating window
    g_temp = g[:, start_p:end_p]
    
    # pass g through both non gating and gating nl
    nongating_letters = nl.torate(g_temp, units=1, bin_rate=bin_rate)
    gating_letters = nl.gating_rate(g_temp, units=1, bin_rate=bin_rate)

    # plot distribution of letters with gating and no gating nls during the gating window
    _plt.close('letter_distribution')
    _plt.figure('letter_distribution')
    _plt.hist([gating_letters.flatten(), nongating_letters.flatten()],bins=10, cumulative=True)

def get_word_info(g, gating_start_t, gating_end_t, letter_length, lettersN, nl, binsN):
    '''
    Comput the information at all time points of the words in both gating/nogating cases.
    Gating starts and ends at gating_start_t and gating_end_t, if words dont overlap this window then gating/nogating is teh same.
    During gating nl.gating_nl will be used and created if it doesn't exist.
    '''
    #_pdb.set_trace()
    gating_start_p = time_to_point(gating_start_t, 0)
    gating_end_p = time_to_point(gating_end_t, 0)
    p0 = gating_start_p
    p1 = gating_end_p

    # pass g through the non gating nl
    #nongating_letters = nl.torate(g, units=1)
    bin_rate = g.max()/binsN
    nongating_letters = nl.torate(g, units=1, bin_rate=bin_rate)
    
    # letters outside the gating window are identical for gating/nongating so when computing gating letters start by copying all nongating ones and then replase letters inside the gating window
    gating_letters = nongating_letters.copy() 
    gating_letters[:, gating_start_p:gating_end_p] = nl.gating_rate(g[:, gating_start_p:gating_end_p], units=1, bin_rate=bin_rate)

    # bin g using percentiles
    bins = list(_np.arange(0, 101, 100/binsN))
    percentiles = _np.percentile(g, bins)
    binned_g = _np.digitize(g.flatten(), percentiles).reshape(g.shape)

    #_plt.figure(num='word_histogram')
    #_plt.hist([binned_g[:,p0:p1], gating_letters[:,p0:p1], nongating_letters[:,p0:p1]], bins=list(range(-5,5)))

    # compute mi between non gating letters and g and between gating letters and g
    nongating_mi = _np.zeros(g.shape[1])
    gating_mi = _np.zeros(g.shape[1])
    for i in range(lettersN-1, len(nongating_mi)):
    #for i in range(100,150):
        #_pdb.set_trace()
        nongating_mi[i] = _info.mi(binned_g[:,i], _info.multi_D_sybmols_to_1D(nongating_letters[:,i-lettersN+1:i+1]))

        # if computing information for a word straddling the information window, computing informaiton from gating_letters, if not just copy the nongating_mi
        if i>=gating_start_p and gating_end_p + lettersN > i:
            gating_mi[i] = _info.mi(binned_g[:,i], _info.multi_D_sybmols_to_1D(gating_letters[:,i-lettersN+1:i+1]))
        else:
            gating_mi[i] = nongating_mi[i]

    nongating_mi.tofile('Data/nongating_word_info_{0}L_{1}ms'.format(lettersN, int(1000*letter_length)))
    gating_mi.tofile('Data/gating_word_info_{0}L_{1}ms'.format(lettersN, int(1000*letter_length)))
    
    
def plot_word_info(lettersN_list=None, length_list=None):
    '''
    get_word_info saves files of the form 'Data/gating_word_info_#L_ms', plot those that match lettersN_list, length_list
    '''

    from os import listdir
    from matplotlib.path import Path
    import matplotlib.patches as patches

    nongating_list = [name for name in listdir('Data/') if name.startswith('nongating_word_info_')]
    gating_list = [name for name in listdir('Data/') if name.startswith('gating_word_info_')]

    _plt.close('word_info')
    fig, ax = _plt.subplots(num='word_info')


    for name in gating_list:
        tokens = name.split("_")
        lettersN = int(tokens[3][:-1])
        length = int(tokens[4][:-2])
        if (lettersN_list is None or lettersN in lettersN_list) and (length_list is None or length in length_list):
            ax.plot(_getTAX(), _np.fromfile('Data/'+name), 'b', label = 'gating', lw=2)
            ax.plot(_getTAX(), _np.fromfile('Data/non'+name), 'r', label = 'no gating', lw=2)

    # add dotted line at 0
    ax.plot([0,0], ax.get_ylim(), ':k', label = '_nolabel_')

    # add a gray rectangle showing gating TW
    verts = [(gating_start_t,0), (gating_start_t,ax.get_ylim()[1]), (gating_end_t, ax.get_ylim()[1]), (gating_end_t, 0), (gating_start_t,0),]
    code = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,]
    path = Path(verts, code)
    patch = patches.PathPatch(path, edgecolor='.75', facecolor='.75', lw=1, alpha=1)
    ax.add_patch(patch)
    ax.set_xlim(-.2, .8)
    ax.set_ylim(-.1, ax.get_ylim()[1])

    #ax.legend(fontsize=10, frameon=False)
    ax.set_xticks((-.2, 0, .2, .6))
    ax.set_xticklabels((-.2, 0, .2, 6), fontsize=10)

    ax.set_yticks((0, .5, 1, 1.5))
    ax.set_yticklabels((0, .5, 1, 1.50), fontsize=10)

    ax.text(0.3, ax.get_ylim()[1]*.60, r'$gating$', color='b')
    ax.text(0.3, ax.get_ylim()[1]*.35, r'$no gating$', color='r')

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Infomration (Bits)', fontsize=10)
    ax.yaxis.set_label_coords(-.25, .4)
    
    # add margin for axis labels
    fig.subplots_adjust(bottom=.35, left=.25, right=1, top=1)

    fig.set_size_inches(2, 1.5)
    fig.savefig('Figures/word_info.pdf')
def _simulate_bipolar(filter_instance):
    pass

def _test_adaptation(contrast_list, filter_instance, nl, adaptation_block):
    _plt.close('adaptation_test')
    fig, ax = _plt.subplots(nrows=3, num='adaptation_test')
    
    #_pdb.set_trace()

    gauss=[]
    ca_concentration=[]
    adapted_output=[]
    labels=[]
    for c in contrast_list:
        # filter some gaussian noise, pass it through a nl and adapt the output
        gauss.append(filter_gaussian_noise(filter_instance, c, samples=10000))

        # In order to get adaptation under "memory_normalization", the "memory" of the signal has to scale with its variance.
        # If for example there is a threshold nonlinearity before adaptation block but the threshold is too low, such that all the signal is in the linear range, then the mean averaged by the memory will be independent of the signal's variance and there will be no effective normalization.
        # The most effective normalization happens when the threshold is right in the middle of the distribution such that half the values are cliped and the mean goes like the standard deviation of the signal
        nl.thresh = gauss[-1].mean()

        ca_concentration.append(nl.torate(gauss[-1]))

        adapted_output.append(adaptation_block.adapt(ca_concentration[-1]))

        labels.append('C={0}%'.format(c))

    bins = 50
    ax[0].hist(gauss, bins=bins, normed=True, alpha=1, label=labels)

    ax[1].hist(ca_concentration, bins=bins, normed=True, alpha=1)

    #return adapted_output
    ax[2].hist(adapted_output, bins=bins, normed=True, alpha=1)

    ax[0].legend()

    return adapted_output



class nonlinear_block:
    def __init__(self, s_type, thresh, units, contrast=None, min_fr=0, max_fr=1, sd=1, slope=1):
        '''
        Init a nonlinear_block object
        Not all imput parameters are used, depending on your choice of 's_type'

        inputs:
            most are self explanatory but...
            
            s_type:         'birect',   usses thresh and slope
                            'sigmoid',  uses min_fr, max_fr, sd, slope

            units:          'linear prediction' or 'sd of linear prediction'
                            if units is 'sd of linear prediction' then before passing a signal through NL the signal has to have SD == 1
                            if units is 'linear prediction' then signal passing through NL can have any SD

        '''
        # s_type can either be 'sigmoid' or 'birect'
        if s_type not in ['sigmoid', 'birect']:
            raise ValueError('s_type has to be either "sigmoid" or "birect"')

        if contrast is not None and contrast>1:
            contrast /=100.0 

        if units not in ['linear prediction' or 'sd of linear prediction']:
            raise ValueError('units has to be either "linear prediction" or "sd of linear prediction"')

        #_pdb.set_trace()
        self.s_type = s_type
        self.units = units
        self.thresh = thresh
        self.contrast = contrast
        self.min_fr = min_fr
        self.max_fr = max_fr
        self.sd  = sd
        self.slope = slope

    def torate(self, linear_prediction, bin_rate=None):
        '''
        pass the linear prediction through the nonlinerity

        units:  0 input is in contrast units. Under ideal adaptation measured nonlinearities at different contrast should overlay each other
                1 input is in light units. Measured nonlinearities at different contrast will not overlay each other.
        
        bin_rate:   if bin_rate is given, responses are discretized by floor(rate/bin_rate)
        '''

        #_pdb.set_trace()

        # store original shape since map works on 1D objects and I will flatten the linear_prediction
        shape_ori = linear_prediction.shape


        # preallocate firing_rate ndarray
        firing_rate = _np.zeros_like(linear_prediction)

        if self.units=='sd of linear prediction':
            raise ValueError('not well implemented.')
            # it seems to me that both the filter, nl and linear prediction should have a 'units' property that they check before interacting with each other.
            thresh = self.thresh*1.0/self.contrast
            slope = self.slope*1.0/self.contrast
            sd = self.sd*1.0/self.contrast
        else:
            thresh = self.thresh
            slope = self.slope
            sd = self.sd

        if sd==0 or slope==0:
            return firing_rate

        #_pdb.set_trace()
        if self.s_type == 'sigmoid':
            from scipy.special import expit
            # pass g through the nonlinearity. I'm using scipy.special.expit which is extremely fast, but requires changing the input according to threshold and sigma
            firing_rate = expit((linear_prediction-thresh)/sd)*self.max_fr + self.min_fr
        elif self.s_type == 'birect':
            '''
            lp1d = linear_prediction.flatten()
            firing_rate = _np.array(list(map(lambda x: 0 if x < thresh else slope*(x - thresh), lp1d))).reshape(shape_ori)
            '''
            firing_rate = (linear_prediction-thresh)*slope
            below_thresh_indices = firing_rate<0
            firing_rate[below_thresh_indices] = 0

        if bin_rate is not None:
            firing_rate = _np.ceil(firing_rate/bin_rate)

        return firing_rate

    def __copy__(self):
        return nonlinear_block(self.s_type, self.thresh, self.units, self.contrast, self.min_fr, self.max_fr, self.sd, self.slope)

    
    """
    def gating_rate(self, linear_prediction, units=0, bin_rate=None):
        '''
        convert linear prediction to firing rate but instead of using the nonlinearity given, a shifted version is used (shift is computed here if not given)
        
        Return the amount by which the threshold shifts. According to my model done in Igor, this is 5 * the SD of the linear prediction during Gaussian stimulation at 3% contrast
        (in igor type "wavestats :contrast0:g1" -> v_sdev = 0.12, and g2 plot shows a threshold shift of .6)
        '''
        #_pdb.set_trace()

        # if self doesn't have a gating_nl, just copy it and shift the threshold
        if not hasattr(self, 'gating_nl'):
            self.gating_nl = self.__copy__()
            filter_instance = filter_block()
            self.gating_nl.thresh -= 5*filter_gaussian_noise(filter_instance, 3).std()     # I want the threshold to be 5*std at 3%

        # now pass the linear prediction through the shifted nonlinearity
        return self.gating_nl.torate(linear_prediction, units=units, bin_rate=bin_rate)
    
    def overwrite_gating_nl(self, new_gating_nl):
        '''
        Overwrite or create self.gating_nl with new_gating_nl
        whether slef.gating_nl exists or not, now new_gating_nl will be used. 
        '''

        self.gating_nl = new_gating_nl

    def _test_gating(self, linear_prediction, units=0, bin_rate=None):
        '''
        return the rate under no gating and gatng for the given linear prediction.
        '''
        _pdb.set_trace()

        lp = linear_prediction.flatten()
        nongating = self.torate(lp, units=units, bin_rate=bin_rate)
        gating = self.gating_rate(lp, units=units, bin_rate=bin_rate)

        _plt.close('test_gating')
        fig, ax = _plt.subplots(num='test_gating')
        ax.hist([lp, nongating, gating], labels=['lp', 'nongating', 'gating'])
        ax.legend()
    """

class adaptation_block:
    '''
    Define an adaptive block, for the time being I'm only implementing dividing by memory+offset
    '''

    def __init__(self, s_type, memory, offset = 0):
        '''
        Input:
        ------
            s_type:     for the time being only "memory_normalization"

            memory:     float, in seconds

            offset:     float, in the same units as the linear prediction

        '''
        self.s_type = "memory_noramlization"
        self.memory = memory
        self.offset = offset

    def adapt(self, signal):
        '''
        signal is probably going to be [Ca]. Divide signal by the result of convolving signal with a decaying exponential with 'memory' and adding an offset

        signal can be 1d array or nd array. But dimension 0 is the dimension to be convolved by the decaying exponential
        '''
        from numpy.linalg import norm

        #_pdb.set_trace()
        # convolve signal with an exponential that decays to 1/e over "memory" seconds
        # time unit in the exponential is the same as in signal, sim_delta_t
        # I'm making the exponential to be long enough such that the last point contributes .01 times what the first point contributes exp(-last/memory)=0.01
        # last = -memory*log(0.01)
        last = -self.memory*_np.log(.01)
        memory_array = _np.exp(-1*_np.arange(0, last, sim_delta_t)/self.memory)
        
        # before convolving make memory_array of unit norm
        memory_array /= norm(memory_array)

        # convolve signal with memory_array. I work on a flatten version of array but first have to Transpose since dim0 corresponds to time
        convolution = _np.convolve(signal.T.flatten(), memory_array)

        # now convolution has len(memory_array) + len(signal.flatten()) points. I'm discarding points from the end to keep convolution the same size as signal. I'm reshaping it to be the original size. From every row now, the first len(memory_array)-1 points are trash because the filter is overlaping signal from different cells. After reshaping to signal's original shape, remove those columns
        convolution = convolution[:-len(memory_array)+1].reshape(signal.shape, order = 'F')

        convolution = _np.delete(convolution, range(len(memory_array)-1), 0)

        signal = _np.delete(signal, range(len(memory_array)-1), 0)

        # divide signal by convolution + offset. Convolution is longer than signal by len(memory_array)-1 points that were added to the front. That's why in the following line I have convolution[len(memory_array)-1:]
        return _np.divide(signal, convolution + self.offset)

class filter_block:
    def __init__(self, size, kernel_path, weight, normed=True):
        '''
        Each filter block represents a decomposable space and time filter.
        For the time being, space is defined as a circular disk and images are filtered with it. If the disk size is 0, no filtering takes place and I will use it for Uniform stimulation. Time is defined through the kernel.

        By combining two of such filters I can accomplish the original simulation where center was one pathway and surround was another pathway, each space-time decomposable that where latter summed.
        I can also add a third pathway that is the peripheral one, with no spatial filter
        '''
        
        self.kernel = _np.fromfile(kernel_path, sep=' ')
        self.size   = size
        self.weight = weight
        self._define_spatial_filter(size*pixperdegree)

    def filter_image(self, image):
        '''
        filter spatial image 'image' with a disk of size self.size
        
        if self.size == 0, no spatial filtering is done and image is returned.

        inputs:
        -------
            image:   2D ndarray with light intensities (probably in the 0-255 range)

        outputs:
        --------
            self.filtered:       filtered image with self.size disk
        '''
        #_pdb.set_trace()
        self.filtered_image = _nd.uniform_filter(image, self.size * pixperdegree, mode='constant')

    def temporal_filter(self, stim):
        '''
        simulate the membrane potential of a cell centered on center = (centerX, centerY) moving according to seq
        
        inputs:
        -------
            stim:       Temporal values of the stimulus prior to kernel filtering.
                        Stim can be the output of spatialy filtering an image and moving it around according to eye movements to generate a temporal stimulus
                        Stim can also be a sequence of intensities as in uniform flickering where no spatial integration takes place.
        
                        for example if self.filter_image(some_image) was called, generating self.filtered_image and a sequence of eye movements 'eye_seq' exists, the following extract the temporal stimulus as seen by a cell centered at 'center' = (centerX, centerY).
                        
                        stim = _np.array([self.filtered_image[seq[0,i]+center[0]][seq[1,i]+center[1]] for i in range(seq.shape[1])])
                        
                        where seq is a 2D ndarray sequence of positions relative to centerX, centerY
                        seq[0][:] are positions in x
                        seq[1][:] are positions in y

        output:
        -------
            mp:         the noiseless membrane potential contribution coming out of this pathway
        '''

        # Filter the center and the surround by its corresponding kernel
        mp = self.weight * _np.convolve(stim, self.kernel, mode='valid')
        
        # combine center and surround
        return mp

    def mean_adapt(image):
        '''
        simulate mean adaptation by forcing image mean to be 127
        '''
        # simulate light adaptation, change image mean to be 127.5
        im *= 127.5/im.mean()

        return image

    def contrast_adapt(image):
        '''
        simulate contrast adaptation by forcing image to be in 0-255 range
        '''
        image = 2**8 * (image-image.min())/(image.max() - image.min())

    def _define_spatial_filter(self, size_in_pixels):
        '''
        Define the spatial filter to be used
        '''

        # make sure that size_in_pixels is int and odd
        size_in_pixels = int(size_in_pixels)

        if size_in_pixels/2==0:
            size_in_pixels+=1

        # create a 2D array of size_in_pixels x size_in_pixels with the weight that each pixel contributes to filtered image.
        half = _np.ceil(size_in_pixels/2)
        y, x = _np.ogrid[-half:half+1, -half:half+1]
        self.spatial_filter = y**2 + x**2 < half**2
        self.spatial_filter = self.spatial_filter / self.spatial_filter.sum()

class cell:
    def __init__(self):
        '''
        all parameters are hardcoded for the time being
        '''

        #_pdb.set_trace()

        # define all pathways needed
        self.center = filter_block(center_size, center_kernel_file, center_weight)

        self.surround = filter_block(surround_size, surround_kernel_file, surround_weight)

        self.periphery = filter_block(periphery_size, periphery_kernel_file, periphery_weight)

        # Define internal threshold
        self.nl = nonlinear_block(nl_type, nl_threshold, nl_units)

        self.adaptation = adaptation_block(adaptation_type, adaptation_memory)

    def filter_image(self, imNumber):
        '''
        load given image and pass it through the spatial component of the filters
        '''
        #_pdb.set_trace()

        image = _loadImage(imNumber)

        self.center.filter_image(image)

        self.surround.filter_image(image)

    def _get_mp_noise_model(self):
        '''
        Simulate the same experiment that Yusuf did to compute the SD of the membrane potential under different gaussian contrasts
        Then, define the sim_mp_noise by keeping the ratio of experimental_mp_nosie/experimental_mp_SD equal to sim_mp_noise/sim_mp_SD

        output:
        -------
            noise_model:        property gets added to the cell object 
        '''
        self.noise_model, _ = reproduce_Yusuf()

        
    def add_mp_noise(self, mp, integration_time):
        '''
        Add noise to the membrane potential 'mp'. 
        
        Algorithm depends heavily on the integrationg_time value:

        integration_time>0:         Noise model is computed from Yusuf's recordings in get_mp_noise_model (which adds property 'noise_model' to cell object)
                                    To compute the sd of the membrane potential a sliding window of length 'integration_time' is used
        integration_time == 0       The SD at each point is taken across cells without combining different times.
                                    mp has to be 2D and mp.std(axis=0) is used

        integration_time == -1      Just compute the SD of mp. A single noise value is used for all mp
        
        if mp is 2D, the 1st points of every row are unreliable. If mp is 1D, the 1st points are unreliable.
        By making mp 1D, computing the running SD and then converting it back to its original shape
        '''
        #_pdb.set_trace()

        # if cell has no attribute noise_model, compute it. This will take a little time since it has to open a file and makes a graph but is only done once
        if not hasattr(self, 'noise_model'):
            self._get_mp_noise_model()

        # Depending on the value of integration_time, compute the SD of mp that is needed in order to generate the random noise
        # Then compute the noise. Noise has to be of the same shape as mp to be added at the end.
        if integration_time > 0:
            sd = _np.zeros_like(mp.flatten())
            
            integration_points = int(integration_time/sim_delta_t)
            for i in range(integration_points, len(sd)):
                sd[i] = mp.flatten()[i-integration_points:i].std()
            
            sd = sd.reshape(mp.shape)

            # in this case sd is already teh same size as mp. Generate an array of noise the same size as mp with a SD of 1 and multiply each value by its corresponding SD
            noise = _np.multiply(_np.random.standard_normal(mp.shape), sd)

        elif integration_time == 0:
            sd = mp.std(axis=0)

            # in this case, sd is 1 row with as many columns as mp
            noise = _np.random.standard_normal(mp.shape) * sd

        elif integration_time ==-1:
            sd = mp.std()

            # in this case, sd is a single value
            noise = _np.random.standard_normal(mp.shape) * sd

        return mp + noise

    def combine_all_mp(self, gating_flag):
        '''
        add the membrane potential contribution coming from center, surround, noise and periphery if gating_flag is set.
        '''
        return self.g + gating_flag * multiply(randn(self.g.shape), self.get_mp_noise)
        

    def filter_gaussian_contrast(self, contrast, samples=1E5):
        '''
        Simulate a gausian experiment. In this case there is no need to filter spatially since stimulus is all the same in space and size of center/surround doesn't affect the output of spatial filter because spatial filtering is normalized by the disk size used
        '''
        stim = fake_gaussian_noise(contrast, samples=samples)


        return self.center.temporal_filter(stim) + self.surround.temporal_filter(stim)

    def processAllImages(self, maxImages=None, maxCellsPerImage=None):
        '''
        Compute the linear prediction of this cell as defined by parameters in self.center, self.surround, when the cell moves over many images from Tkacik's data base
        Cells are moving according to a FEM + a saccade that happens at time 0.
        Time in the simulation is defined by sim_start_t, sim_end_t and sim_delta_t, the time axis is tax = arange(sim_start_t, sim_end_t, sim_delta_t)

        inputs:
        -------
            maxImages:          integer, optional parameter defining the maximum number of images to use
                                defaults to None, meaning use all images
        
        outpus:
        -------
            g:                  2D ndarray, the linear prediction of many identical cells over many images
                                g[i][:]     is the linear prediction of cell i over time
                                g[:][t0]    is the linear prediction of all cells and all images at time t0
        '''
        #_pdb.set_trace()

        # try loading 'LinearPrediction' from Data/, if that fails, compute it
        from os import listdir
        if 'LinearPrediction' in listdir('Data/'):
            g = _np.fromfile('Data/LinearPrediction').reshape(-1,300)
            return g

        if images_list is None:
            _getImagesPath()
        

        # estimate number of cells per image
        if maxCellsPerImage is None:
            centerD = self.center.size*pixperdegree
            imSize = _loadImage(0).shape
            maxCellsPerImage = _np.floor(imSize[0]/centerD)*_np.floor(imSize[1]/centerD)

        # compute time axis of simulation
        tax = _getTAX()

        # preallocate array for all linear predictions
        g = _np.zeros((maxCellsPerImage*len(images_list), len(tax)))
    
        #_pdb.set_trace()
        nextCell = 0
        for imNumber in range(len(images_list)):
            if imNumber == maxImages:
                break
            
            print(images_list[imNumber])
            t = _time()
            nextCell = self._processOneImage(imNumber, g, nextCell, maxCellsPerImage)
            print('\t{0} cells processed in {1} secs'.format(nextCell, _time()-t))

        g = g[:nextCell][:]
        g.tofile('Data/LinearPrediction')

        return g

    def _processOneImage(self, imNumber, g, nextCell, maxCells=None):
        '''
        Compute the linear prediction of several instances of these cell moving over the image described by imNumber

        inputs:
        -------
            imNumber:   integer, image to load from images_list
        
            g:          2D array with all inear predictions. Will be modified in place

            nextCell:   index into the 1st dimension of g where next simulated cell should be incorporated.
                        
            maxCells:   int, optional. If given limits how many cells will be processed on a given image.

        output:
            g:          modified in place, incorporates the linear predictions from image imNumber in g, starting from 
                        row = nextCell
        '''

        
        # filter image with center and surround spatial filters. Property 'filtered_image' is set in each filter_block
        self.filter_image(imNumber)

        # grab the eye movement sequence
        seq = _getEyeSeq(len(self.center.kernel))

        # grab non overlapping cells from image such that when moved according to seq, they are always whithing the boundaries
        centerD     = int(self.center.size * pixperdegree)    # center's diameter in pixels
        surroundD   = int(self.surround.size * pixperdegree)     # surround's diameter in pixels

        #_pdb.set_trace()
        image_size = self.center.filtered_image.shape

        startX  = int(_np.ceil(surroundD - min(seq[0][:])))
        endX    = int(_np.floor(image_size[0] - surroundD - max(seq[0][:])))
        startY  = int(_np.ceil(surroundD - min(seq[1][:])))
        endY    = int(_np.floor(image_size[1] - surroundD - max(seq[1][:])))
        
        i = 0
        #_pdb.set_trace()
        for center in _product(range(startX, endX, centerD), range(startY, endY, centerD)):
            # extract from filtered versions of image the time series corresponding to central and surround contributions of this particular cell
            center_stim  = _np.array([self.center.filtered_image[seq[0,i]+center[0]][seq[1,i]+center[1]] for i in range(seq.shape[1])])
            surround_stim = _np.array([self.surround.filtered_image[seq[0,i]+center[0]][seq[1,i]+center[1]] for i in range(seq.shape[1])])

            # pass those time series through the temporal filter and combine them
            g[nextCell, :] = self.center.temporal_filter(center_stim) + self.surround.temporal_filter(surround_stim)
            
            nextCell += 1
            i+=1
            if i==maxCells:
                return nextCell

        return nextCell

    def get_gating_contribution(self):
        '''
        compute the contribution of gating to the membrane potential
        '''

        gating_mp = _np.zeros_like(_getTAX())

        saccade_point = int(-sim_start_t/sim_delta_t)

        filter_length = len(self.periphery.kernel)

        gating_mp[saccade_point:saccade_point+filter_length] = self.periphery.kernel

        return gating_mp

    def get_ca_concentration(self):
        self.nl.torate

    def compare_gaussian_to_simulation(self, g, contrast, times, nls):
        '''
        This might not need to be a cell method but too lazzy to change it.

        Plots g values at the given times (might be empty)

        Plots filtered gaussian stimulus by cell 'self'. contrast can be one float or a list, can be empty list as well

        Plots also all nonlinear objects in nls. Can be an empty list.
        '''
        #_pdb.set_trace()
        if not _np.iterable(contrast):
            contrast = [contrast]

        if not _np.iterable(times):
            times = [times]

        if not _np.iterable(nls):
            nls = [nls]

        _plt.close('UFlicker_sim_comparisson')
        fig, ax1, = _plt.subplots(num='UFlicker_sim_comparisson')
        
        bins = 200
        # simulation data
        for t in times:
            point = time_to_point(t,0)
            data_to_hist = g[:,point]
        
            label = r'$t ={0: G}ms$'.format(int(1000*t))
            hist, bins, patches = ax1.hist(data_to_hist, bins=bins, normed=True, histtype='stepfilled', alpha=.5, label=label)
        
        # Gaussian, like in UFLicker
        for C in contrast:
            gaussian_g = self.filter_gaussian_contrast(C)
            ax1.hist(gaussian_g, bins = bins, histtype='stepfilled', normed=True, alpha=.5, label='{0}% Gaussian'.format(C))

        if len(nls):
            ax2 = ax1.twinx()
        
        for i, nl in enumerate(nls):
            if len(nls)==2 and i==0:
                label = r'$basal$'
            elif len(nls)==2 and i==1:
                label = r'$gating$'
            else:
                label = '_nolegend_'

            ax2.plot(bins, nl.torate(bins), lw=2, label=label)

        ax1.legend()
        if len(nls):
            ax2.legend(loc='upper left')

        fig.savefig('Figures/UFlicker_sim_comparisson')
        return fig, ax1


    def _test_filtering(self):
        '''
        Test that filter_gaussian_contrast (avoiding spatial filtering) gives the same result had I used the spatial filtering.
        
        generate a mean intensity image of 127 and filter it with both center and surround, then generate stim by concatenating those values and pass them to temporal filtering

        On the other hand, send to filter_gaussian_contrast a stim with 0 contrast
        '''
        sim1 = self.filter_gaussian_contrast(0)

        image = _loadImage(0)
        image_1 = _np.ones_like(image)*127
        self.center.filter_image(image_1)
        self.surround.filter_image(image_1)

        sim2 = self.center.temporal_filter([self.center.filtered_image[500,500]]*1000) + self.surround.temporal_filter([self.surround.filtered_image[500,500]]*1000)

        return sim1, sim2
