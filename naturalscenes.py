'''
naturalscenes.py

A module to load and process natural scenes from Tkacik database
'''
import numpy as _np
from scipy import ndimage as _nd
from glob import glob as _glob
import matplotlib.pyplot as _plt
import pdb as _pdb

pixperdegree = 46       # converts pixels in image to degrees
imList = ''             # will be computed later on
centerK = _np.array([]) # these are holders for kernels to be loaded
surroundK = _np.array([])
clipK = 117             # point at which kernels are clipped
deltaT = .005           # time resolution of kernels in seconds
rwStep = .01            # in degrees
saccadeSize = 6         # in degrees
centerS = 1             # in degrees
surroundS = 2.5         # scale factor, surround is surroundS bigger than centerS
surroundW = 1           # weight of surround pathway relative to center one (center pathway has weight=1)
startT = -.2            # in seconds
endT = 2                # in seconds

def getImagesPath(path=None):
    if path is None:
        path = '/Users/jadz/Documents/Notebook/Matlab/Natural Images DB/RawData/*/*LUM.mat'
        
    global imList
    imList = _glob(path)

def loadImage(imNumber):
    '''
    Load an image from the database

    inputs:
    -------
        imNumber:   integer, specifying which element from imList to load

    output:
        image:      ndarray with the image
    '''
    from scipy import io

    if imList == '':
        getImagesPath()

    return io.loadmat(imList[imNumber])['LUM_Image']

def filterImage(imNumber, size):
    '''
    filter image (imList[imNumber]) with a disc of size 'size' degrees

    inputs:
    -------
        imNumber:   integer, what image from imList to process

        size:   float, is in degrees
                actual pixels are computed from piperdegree
    outputs:
    --------
        filtered:   returns the filtered image
    '''

    im = loadImage(imNumber)
    return _nd.uniform_filter(im, size*pixperdegree, mode='constant')

def loadKernels(path=None):
    if path is None:
        path = '/Users/jadz/Documents/Notebook/Experiments/Simulations/Natural Images DB/New Saccade analysis/'
    
    global centerK, surroundK
    centerK = _np.genfromtxt(path+'centerKernel.txt')
    surroundK = _np.genfromtxt(path+'surroundKernel.txt')

    centerK = centerK[:clipK]
    surroundK = surroundK[:clipK]

def getEyeSeq():
    '''
    Generate a sequence of eye movements in both x and y directions
    The sequence is a 2D ndarray compossed of steps. 
    seq[0][p] is the step in the x direction at point p
    seq[1][p] is the step in the y direction at point p

    seq starts at time startT-( len(centerK)+1 ) * deltaT and ends at time endT
    in this way, when convolving with mode='valid' the output will have samples spanning startT and endT

    output:
    -------
        seq:    2D ndarray with steps
    '''
    #_pdb.set_trace()
    # figure out how many time points sequence has
    if len(centerK)==0:
        loadKernels()
        
    stepsN = (endT-startT)/deltaT + len(centerK) - 1

    # generate the FEM part of the sequence
    seq = _np.random.randn(2, stepsN)
    seq *= pixperdegree*rwStep

    # add saccade in both x and y for the time being
    saccadePnt = len(centerK)-1 - startT/deltaT
    seq[:,saccadePnt]+=saccadeSize*pixperdegree

    # change from steps to actual positions
    seq = seq.cumsum(1)

    return seq.astype('int16')

def getOneLP(imNumber, centerX, centerY):
    '''
    simulate the linear prediction (g) for one cell centered on (centerX, centerY) in imNumber moving according to getEyeSeq()
    
    inputs:
    -------
        imNumber:   integer, image to load from imList

        centerX/Y:  integer, pixel describing the cell's center postion

    output:
    -------
        g:          ndarray, the linear prediction 
    '''

    # filter image with center and surround spatial filters. filterImage takes inputs in degrees, not pixels
    imC = filterImage(imNumber, centerS)
    imS = filterImage(imNumber, centerS*surroundS)

    # grab the eye movement sequence
    seq = getEyeSeq()

    # for the given cell center, extract the sequence of luminance values as seen by the center and the surround
    centerG = _np.array([imC[seq[0,i]][seq[1,i]] for i in range(seq.shape[1])])
    surroundG = _np.array([imS[seq[0,i]][seq[1,i]] for i in range(seq.shape[1])])

    # Filter the center and the surround by its corresponding kernel
    if len(centerK)==0 or len(surroundK)==0:
        loadKernels()

    centerG = _np.convolve(centerG, centerK, mode='valid')
    surroundG = _np.convolve(surroundG, surroundK, mode='valid')
    
    g = centerG + surroundW*surroundG

    plot(g, centerG, surroundG)

def plot(*argv):
    tax = _np.array(_np.arange(startT, endT, deltaT))
    
    _plt.close('all')
    _plt.figure('g')
    for arg in argv:
        _plt.plot(tax, arg)
