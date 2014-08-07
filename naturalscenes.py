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

d={}
d['pixperdegree'] = 46       # converts pixels in image to degrees
d['clipK'] = 117             # point at which kernels are clipped
d['deltaT'] = .005           # time resolution of kernels in seconds
d['rwStep'] = .001            # in degrees
d['saccadeSize'] = 6         # in degrees
d['centerS'] = 1             # diameter, in degrees
d['surroundS'] = 2.5         # scale factor, surround is d['surroundS'] bigger than d['centerS']
d['surroundW'] = 1           # weight of surround pathway relative to center one (center pathway has weight=1)
d['startT'] = -2             # in seconds
d['endT'] = 2                # in seconds

def _getImagesPath(path=None):
    if path is None:
        path = '/Users/jadz/Documents/Notebook/Matlab/Natural Images DB/RawData/*/*LUM.mat'
        
    global d
    d['imList'] = _glob(path)

def _loadImage(imNumber):
    '''
    Load an image from the database

    inputs:
    -------
        imNumber:   integer, specifying which element from d['imList'] to load

    output:
        image:      ndarray with the image
    '''
    from scipy import io
    if 'imList' not in d:
        _getImagesPath()

    return io.loadmat(d['imList'][imNumber])['LUM_Image']

def _filterImage(imNumber, size):
    '''
    filter image (d['imList'][imNumber]) with a disc of size 'size' degrees

    inputs:
    -------
        imNumber:   integer, what image from d['imList'] to process

        size:   float, is in degrees
                actual pixels are computed from piperdegree
    outputs:
    --------
        filtered:   returns the filtered image
    '''

    im = _loadImage(imNumber)
    return _nd.uniform_filter(im, size*d['pixperdegree'], mode='constant')

def _loadKernels(path=None):
    if path is None:
        path = '/Users/jadz/Documents/Notebook/Experiments/Simulations/Natural Images DB/New Saccade analysis/'
    d['centerK'] = _np.genfromtxt(path+'centerKernel.txt')
    d['surroundK'] = _np.genfromtxt(path+'surroundKernel.txt')
    d['centerK'] = d['centerK'][:d['clipK']]
    d['surroundK'] = d['surroundK'][:d['clipK']]

def _getEyeSeq():
    '''
    Generate a sequence of eye movements in both x and y directions
    The sequence is a 2D ndarray compossed of steps. 
    seq[0][p] is the step in the x direction at point p
    seq[1][p] is the step in the y direction at point p

    seq starts at time d['startT']-( len(centerK)+1 ) * d['deltaT'] and ends at time d['endT']
    in this way, when convolving with mode='valid' the output will have samples spanning d['startT'] and d['endT']

    output:
    -------
        seq:    2D ndarray with steps
    '''
    #_pdb.set_trace()
    # figure out how many time points sequence has
    if 'centerK' not in d:
        _loadKernels()
        
    stepsN = (d['endT']-d['startT'])/d['deltaT'] + len(d['centerK']) - 1

    # generate the FEM part of the sequence
    seq = _np.random.randn(2, stepsN)
    seq *= d['pixperdegree']*d['rwStep']

    # add saccade in both x and y for the time being
    saccadePnt = len(d['centerK'])-1 - d['startT']/d['deltaT']
    seq[:,saccadePnt]+=d['saccadeSize']*d['pixperdegree']

    # change from steps to actual positions
    seq = seq.cumsum(1)

    return seq.astype('int16')

def processAllImages(maxImages=None, maxCellsPerImage=None):
    '''
    Compute the linear prediction of a cell as defined by parameters in d moving over many images from Tkacik's data base
    Cells are moving according to a FEM + a saccade that happens at time 0.
    Time in the simulation is defined by d['startT'], d['endT'] and d['deltaT'], the time axis is tax = arange(d['startT'], d['endT'], d['deltaT'])

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
    if 'imList' not in d:
        _getImagesPath()
    
    #_pdb.set_trace()

    # estimate number of cells per image
    if maxCellsPerImage is None:
        centerD = d['centerS']*d['pixperdegree']
        imSize = _loadImage(0).shape
        maxCellsPerImage = _np.floor(imSize[0]/centerD)*_np.floor(imSize[1]/centerD)

    # compute time axis of g
    tax = _np.arange(d['startT'], d['endT'], d['deltaT'])

    # preallocate array for all linear predictions
    g = _np.empty((maxCellsPerImage*len(d['imList']), len(tax)))
   
    #_pdb.set_trace()
    nextCell = 0
    for imNumber in range(len(d['imList'])):
        if imNumber == maxImages:
            break
        
        print(d['imList'][imNumber])
        nextCell = _processOneImage(imNumber, g, nextCell, maxCellsPerImage)
        print('\t{0} cells processed'.format(nextCell))

    g = g[:nextCell][:]
    return g

def _processOneImage(imNumber, g, nextCell, maxCells=None):
    '''
    Compute the linear prediction of several cells moving over the image described by imNumber

    inputs:
    -------
        imNumber:   integer, image to load from d['imList']
    
        g:          2D array with all inear predictions. Will be modified in place

        nextCell:   index into the 1st dimension of g where next simulated cell should be incorporated.
                    
        maxCells:   int, optional. If given limits how many cells will be processed on a given image.

    output:
        g:          modified in place, incorporates the linear predictions from image imNumber in g, starting from 
                    row = nextCell
    '''

    # filter image with center and surround spatial filters. filterImage takes inputs in degrees, not pixels
    imC = _filterImage(imNumber, d['centerS'])
    imS = _filterImage(imNumber, d['centerS']*d['surroundS'])

    # grab the eye movement sequence
    seq = _getEyeSeq()

    # grab non overlapping cells from image such that when moved according to seq, they are always whithing the boundaries
    centerD = d['centerS']/2.0*d['pixperdegree']    # center's diameter in pixels
    surroundD = d['surroundS']*centerD        # surround's diameter in pixels

    startX = _np.ceil(surroundD - min(seq[0][:]))
    endX = _np.floor(imC.shape[0] - surroundD - max(seq[0][:]))
    startY = _np.ceil(surroundD - min(seq[1][:]))
    endY = _np.floor(imC.shape[1] - surroundD - max(seq[1][:]))
    
    i = 0
    for center in _product(_np.arange(startX, endX, centerD), _np.arange(startY, endY, centerD)):
        _getOneLP(imC, imS, seq, center, g, nextCell)
        nextCell += 1
        i+=1
        if i==maxCells:
            return nextCell

    return nextCell

def _getTAX():
    if 'tax' not in d:
        d['tax'] = _np.arange(d['startT'], d['endT'], d['deltaT'])

def _getOneLP(centerIm, surroundIm, seq, center, g, nextCell):
    '''
    simulate the linear prediction (g) for one cell centered on center = (centerX, centerY) in imNumber moving according to getEyeSeq()
    
    inputs:
    -------
        centerIm:   2D ndarray, image to be processed after filtering with center spatial filter

        surroundIm: 2D ndarray, image to be processed after filtering with surround spatial filter

        seq:        2D ndarray, sequence of positions relative to centerX, centerY
                    seq[0][:] are positions in x
                    seq[1][:] are positions in y

        center:     tuple with 2 integers, (x, y) describing the cell's center postion in the image

        g:          2D ndarray, holds all linear predictions and is modified in place

        nextCell:   index into g where next linear prediction will be inserted
        
    output:
    -------
        g:          ndarray, the linear prediction 
    '''

    # for the given cell center, extract the sequence of luminance values as seen by the center and the surround
    centerG = _np.array([centerIm[seq[0,i]+center[0]][seq[1,i]+center[1]] for i in range(seq.shape[1])])
    surroundG = _np.array([surroundIm[seq[0,i]+center[0]][seq[1,i]+center[1]] for i in range(seq.shape[1])])

    # Filter the center and the surround by its corresponding kernel
    if 'centerK' not in d or 'surroundK' not in d:
        _loadKernels()

    centerG = _np.convolve(centerG, d['centerK'], mode='valid')
    surroundG = _np.convolve(surroundG, d['surroundK'], mode='valid')
    
    #_pdb.set_trace()
    g[nextCell,:] = centerG + d['surroundW']*surroundG
    

def sampleG(g, num):
    '''
    make a plot with 'num' random cells
    '''
    if 'tax' not in d:
        _getTAX()
    
    _plt.close('Sample G')
    _plt.figure('Sample G')
    for i in range(num):
        index = _np.random.randint(0, g.shape[0])
        _plt.plot(d['tax'], g[index,:])

    _plt.xlabel('Time (s)')
    _plt.ylabel('Linear Prediction (AU)')

def estimateEntropy(covG):
    '''
    From the 2D ndarray g with the linear prediciton of many cells, estimate the entropy of the signal as a function of time
    
    Implementaiton note:
        I changed the reuse code for jointEntropy, in this way is easier to maintain

    input:
    ------
        g:          2D ndarray with linear predictions
                    1st dimension is cells
                    2nd dimension is time, spanning from d['startT'] to d['endT'] in steps of d['deltaT']

    output:
    -------
        entropy:    1D ndarray spanning the same time as g with the entropy at each timepoint.
    '''
    
    return = _np.array([jointEntropy(covG, _np.array([p], dtype=int)) for p in range(covG.shape[0])])

def jointEntropy(covG, points):
    '''
    Compute the joint entropy of the linear prediction g (associated with covariance matrix covG) at the points along the time axis given by 'points'. 
    Namely, compute:
    H(g(p0), g(p1), ..., g(pn)) where p0, p1, ... pn are in 'points'

    inputs:
    -------
        covG:       2D ndarray, the covariance matrix of the linear prediction

        points:     ndarray of ints, the points along the time axis to extract a sub matrix of covG and compute the joint entropy
    '''

    if not isinstance(points, _np.ndarray):
        raise TypeError('points in jointEntropy should be ndarray')

    if not issubclass(points.dtype.type, _np.integer):
        raise TypeError('points in jointEntropy should be ndarray of integers')

    # compute an array with the indeces corresponding to condTimes
    #condTimesArray = _np.floor((_np.array(condTimes) - d['startT'])/d['deltaT'])
    
    # extract the sub array from covG corresponding to condTimesArray. I will take elements from a flatten version of covG. The function to take the elements form a 1D array is 'take' and I'm using product form itertools to get all combinations of the indexes in condTimesArray. At the end I'm reshaping it to be a square matrix with each dimension having len(points) elements
    from itertools import product
    subCovG = _np.take(covG.flatten(), [i[0]+ covG.shape[0]*i[1] for i in product(points, points)]).reshape(-1, len(points))
    
    # now compute and return the entropy
    return _info.gaussianEntropy(subCovG)

def condEntropy(covG, condTimes):
    '''
    Compute the conditional entropy of the linear prediction g (with covariance matrix 'covG') at every point along the time axis conditional on the linear prediction at all previous times in condTimes
    Computes:
        H(g(t) | g(t0), g(t1), ..., g(tn)) where t0, t1, ... tn are in condTimes
    
    Implementation notes:
        H(X | Y) = H(X, Y) - H(Y)
    
    inputs:
    -------
        covG:       2D ndarray, the covariance matrix associated with the linear prediction g

        condTimes:  an iterable of floats, all times should be >0

    '''


    # change condTimes by condPoints, which are the points into the time axis of covG pointed by condTImes
    condPoints = _np.array([_np.floor((t)/d['deltaT']) for t in condTimes], dtype=_np.integer)

    if _np.any(condPoints)<=0:
        raise ValueError('condEntropy requires condTimes to have all values that are >0')

    maxDelay = max(condPoints)
    
    # compute H(Y) in the formula above
    H_Y = _np.array([jointEntropy(covG, p-condPoints) if p>=maxDelay else 0 for p in range(covG.shape[0])])
    
    # compute H(X,Y) in the formula above
    condPoints = _np.array([0] + [i for i in condPoints])
    H_XY = _np.array([jointEntropy(covG, p-condPoints) if p>=maxDelay else 0 for p in range(covG.shape[0])])

    return H_XY - H_Y


def process4paper(g=None):

    # this will take some hours, processing over 300 images
    if g is None:
        g = processAllImages()

    # form the array of linear predicitons, compute the covariance matrix
    covG = _np.cov(g, rowvar=0)

    # Compute entropy as a function of time
    entropy = estimateEntropy(covG)

    # Compute the MI between points separated by 50ms
    condEntropy_1 = condEntropy(covG, [.05])
    condEntropy_2 = condEntropy(covG, [.05, .1])
    condEntropy_3 = condEntropy(covG, [.05, .1, .15])
   
    # start making figures
    # Sample G
    sampleG(g, 100)
    _plt.savefig('Figures/sampleG')
    _plt.xlim(-1,1)

    # Entropy and cond Entropy
    if 'tax' not in d:
        _getTAX()

    _plt.close('MI')
    _plt.figure('MI')
    _plt.plot(d['tax'], entropy, linewidth=2, label='H(g(t))')
    _plt.plot(d['tax'], condEntropy_1, linewidth=2, label='H(g(t) | g(t-50ms))')
    _plt.plot(d['tax'], condEntropy_2, linewidth=2, label='H(g(t) | g(t-50ms), g(t-100ms))')
    _plt.plot(d['tax'], condEntropy_3, linewidth=2, label='H(g(t) | g(t-50ms), g(t-100ms),g(t-150ms)')
    _plt.legend(loc='lower right')
    _plt.xlabel('Time (s)')
    _plt.ylabel('Bits')
    _plt.title('Entropy')
    _plt.xlim(-1, 1)
    _plt.savefig('Figures/Entropy')
