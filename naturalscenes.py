'''
naturalscenes.py

A module to load and process natural scenes from Tkacik database
'''
import numpy as _np
from scipy import ndimage as _nd
from glob import glob as _glob
from itertools import product as _product
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
centerS = 1             # diameter, in degrees
surroundS = 2.5         # scale factor, surround is surroundS bigger than centerS
surroundW = 1           # weight of surround pathway relative to center one (center pathway has weight=1)
startT = -2             # in seconds
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

def processAllImages():
    if imList == '':
        getImagesPath()

    # estimate number of cells per image
    surroundD = surroundS*centerS*pixperdegree
    imSize = loadImage(0).shape
    cellsN = _np.floor(imSize[0]/surroundD)*_np.floor(imSize[1]/surroundD)

    # compute time axis of g
    tax = _np.arange(startT, endT, deltaT)

    # preallocate array for all linear predictions
    g = _np.empty((cellsN*len(imList), len(tax)))
   
    #_pdb.set_trace()
    nextCell = 0
    for imNumber in range(len(imList)):
        print(imList[imNumber])
        nextCell = processOneImage(imNumber, g, nextCell)
        
        if imNumber==10:
            break
    
    g = g[:nextCell][:]
    return g

def processOneImage(imNumber, g, nextCell):
    '''
    Compute the linear prediction of several cells moving over the image described by imNumber

    inputs:
    -------
        imNumber:   integer, image to load from imList
    
        g:          2D array with all inear predictions. Will be modified in place

        nextCell:    index into the 1st dimension of g where next simulated cell should be incorporated.
                    
    output:
        g:          modified in place, incorporates the linear predictions from image imNumber in g, starting from 
                    row = nextCell
    '''

    # filter image with center and surround spatial filters. filterImage takes inputs in degrees, not pixels
    imC = filterImage(imNumber, centerS)
    imS = filterImage(imNumber, centerS*surroundS)

    # grab the eye movement sequence
    seq = getEyeSeq()

    # grab non overlapping cells from image such that when moved according to seq, they are always whithing the boundaries
    centerD = centerS/2*pixperdegree    # center's diameter in pixels
    surroundD = surroundS*centerD        # surround's diameter in pixels

    startX = _np.ceil(surroundD - min(seq[0][:]))
    endX = _np.floor(imC.shape[0] - surroundD - max(seq[0][:]))
    startY = _np.ceil(surroundD - min(seq[1][:]))
    endY = _np.floor(imC.shape[1] - surroundD - max(seq[1][:]))
    
    i = 0
    for center in _product(_np.arange(startX, endX, centerD), _np.arange(startY, endY, centerD)):
        #print('processing image centered on: {0}'.format(center))
        #g[nextCell, :] = getOneLP(imC, imS, seq, center)#, g, nextCell)
        getOneLP(imC, imS, seq, center, g, nextCell)
        nextCell += 1
        i+=1
        if i==2:
            return nextCell

    return nextCell

def getOneLP(centerIm, surroundIm, seq, center, g, nextCell):
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
    if len(centerK)==0 or len(surroundK)==0:
        loadKernels()

    centerG = _np.convolve(centerG, centerK, mode='valid')
    surroundG = _np.convolve(surroundG, surroundK, mode='valid')
    
    #_pdb.set_trace()
    g[nextCell,:] = centerG + surroundW*surroundG
    

def sampleG(g, num):
    x = _np.arange(startT, endT, deltaT)
    print(x.shape, g.shape)
    _plt.close('Sample G')
    _plt.figure('Sample G')
    for i in range(num):
        index = _np.random.randint(0, g.shape[0])
        _plt.plot(x, g[index,:])
