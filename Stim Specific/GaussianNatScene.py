#!/Users/jadz/anaconda/bin/python
import experiment as exp
import basicAnalysis as ba
import filtertools as ft
import stimulustools as st
import responsetools as rt
import numpy as _np
import pdb as _pdb
import matplotlib.pyplot as _plt

def preProcess(expName, spikeFile):
    #_pdb.set_trace()
    # load exp variables
    d1 = exp.loadVariables(spikeFile)

    # add any previously used parameters from parameters.txt
    d1.update(exp.loadParameters(expName))

    # If first time running analysis on this path, parameters.txt will not have some needed variables and therefore will fail at loading them. Use defaults
    if 'fixationLength' not in d1.keys():
        d1['fixationLength'] = 2*d1['PDperiod']
    if 'saccadesN' not in d1.keys():
        d1['saccadesN'] = int(_np.round(d1['endT']/d1['fixationLength']))
    if 'expName' not in d1.keys():
        d1['expName'] = expName
    if 'waitframes' not in d1.keys():
        d1['waitframes'] = 3

    exp.writeParameters(d1)

    # load all cells
    cells = exp.loadAllCells(spikeFile)

    # Load the sequence of home and targets
    blockSeq = Load_HT_Seq(d1['saccadesN'])

    # Get each start/end time per block
    blockStartT = [d1['fixationLength']*i for i in range(d1['saccadesN'])]
    blockEndT = [blockStartT[i]+d1['fixationLength'] for i in range(d1['saccadesN'])]

    # with blockSeq, I can compute number of targets (half conditions are home and half are targets) and the trials per condition. 
    # I have to exclude the 1st element from blockSeq because it is 'None'
    d1['targetN'] = int((max(blockSeq[1:])+1)/2)
    d1['trialsN'] = round(d1['endT']/d1['fixationLength']/d1['targetN']/2)
    
    return d1, cells, blockSeq, blockStartT, blockEndT

def divideSpikesPerCondition(d1, cells, blockSeq, blockStartT, blockEndT):

    # loop through all cells and divide spikes for each cell according to corresponding H or T        
    cells = [exp.divideSpikes(cell, blockStartT, blockEndT, blockSeq, 0) for cell in cells] 

    # now cells is a list where each element (corresponding to a single cell) is a list of 16 different conditions.
    # rearrange each cell into a 2D list such that conditions 0-7 are in the 1st col and conditions 8-15 into the 2nd col.
    # rows are the first index into the cell list
    # cols are indices into the rows (or the 2nd index in the cell)
    cells = [[[cell[i+d1['targetN']*j] for j in range(2)] for i in range(d1['targetN'])] for cell in cells]
    return d1, cells

def processSTA(expName='GNS', spikeFile='GaussianNatScene.spk', dnew={}):
    '''
    Load spikes from spikeFile
    Extract last parameters used from expName in parameters.txt
    Update any parameters according to dnew (in case I want to change something)

    do whatever processing I feel is necessary
    '''
    # load basic dictionary with experimental data and all cells
    print('Loading cells and variables')
    d, cells = divideSpikesPerCondition(*preProcess(expName, spikeFile))
    d.update(dnew)

    # compute PSTHs
    psths = ba.processNested(ba.psth, 0, cells, d['fixationLength'], d['trialsN'], returnFlag=1)
    psthsX = ba.psth(cells[0][0][0], d['fixationLength'], d['trialsN'], returnFlag=2)

    # load Gaussian noise wave to be able to correlate
    print('Loading noise')
    noise = loadNoise(d, (2))

    # create stimulus object
    stim = st.stimulus(noise, d=d)
    
    # upsample stim to about 5ms time bins
    stim = stim.upsamplestim(int(stim.deltaT/.005))

    # compute STA for each condition
    print('Computing stas')
    stas = ba.processNested(ft.getsta, 2, stim.tsa, stim.stim, cells, 100, returnFlag = 1)
    
    # compute one x axis for all stas
    print('Computing stas X axis')
    stasX = ft.getsta(stim.tsa, stim.stim, cells[0][0][0], 100, returnFlag=2)

    # since in GNS filter sign is irrelevant, flip them to have negative peaks (off like filters)
    stas = ba.processNested(flipSTA, 0, stas)

    # since on/off filters are somewhat arbitrary with these stimulus, force them all to be of the same type (peak being off)
    #return d, cells, psths, psthsX
    return d, stim, cells, psths, psthsX, stas, stasX, stim

def processInformation(expName='GNS', spikeFile='GaussianNatScene.spk', dnew={}):
    '''
    Load spikes from spikeFile
    Extract last parameters used from expName in parameters.txt
    Update any parameters according to dnew (in case I want to change something)

    do whatever processing I feel is necessary
    '''
    # load basic dictionary with experimental data, cells and the seq of Home and Targets
    print('Loading cells and variables')
    d, cells, blockSeq, blockStartT, blockEndT = preProcess(expName, spikeFile)
    d.update(dnew)

    # Define a TW for computing basic response properties
    TW_startT = .05
    TW_endT = .20
    binsN = 4
    eventStartT = _np.array(blockStartT) + TW_startT
    eventEndT = _np.array(blockStartT) + TW_endT
    binLength = (TW_endT - TW_startT)/binsN

    # compute Latency and spike count
    print('Processing Latency for all cells')
    latency = ba.processNested(rt.getLatency, 0, cells, eventStartT, eventEndT, noSpikeSymbol = - binLength/2)
    print('Processing Spike Count for all cells')
    spkCnt = ba.processNested(rt.getSpkCnt, 0, cells, eventStartT, eventEndT)

    # convert latency into discrete symbols (for each cell)
    latency = ba.processNested(lambda x: _np.floor(x/binLength), 0, latency)
    
    
    return d, cells, latency, spkCnt, blockSeq

    
def Load_HT_Seq(N):
    '''
    Load the Home and Target sequence
    For each presentation, Targets are randomized and then interleaved with Home such that the sequence is: H T[Seq[0]] H T[Seq[1]] H ... T[Seq[n-1]]

    inputs:
    -------
        N:  total number of elements in the output sequence. This is not how many Targets or Homes I have in a presentation, but the (number of saccades per presentation) * (number of presentations)
    
    output:
    -------
        seq:    the sequence of stim IDs len(seq)*d['fixationLength']=d['endT']
                in seq, each element is an integer from 0 to 2*d['targetN']
                if seq[i]:  is in range(d['targetN'])                        seq[i] is a saccade from H to target identity seq[i]
                if seq[i]:  is in range(d['targetN'], 2*d['targetN'])        seq[i] is a saccade from Target identity seq[i]-d['targetN'] to H
    '''
    # Load sequence of targets (T, randomized). In version 1 of the experiment, the stimulus intercalates H in between targets starting from H
    # Final sequence is H,T0,H,T1,H, ...
    with open('/Users/jadz/Documents/Notebook/Design and Analysis/GaussianNatScene/TargetSequence.bin', 'rb') as f_in:
        T = [c-1 for c in f_in.read()]

    # I'm going to intercalate H in target sequence. Although there is only one H in this version, I want to distinguish Hs based on the previous target. 
    # There are 'tartetsN' total targets and therefore 2*targetsN identifiers in the sequence.
    # The stim H->T[i] gets assigned identifier T[i]
    # The stim T[i]->H gets assigned identifier targetsN + T[i]
    # The first H comes from a gray screen, I am assigning it to None latter on
    # the list comprehension below starts from the saccade from that first H to the 1st target
    targetsN = _np.array(T).max()+1
    seq = [T[i//2] if i%2==0 else T[i//2]+targetsN for i in range(N-1)]
    
    seq.insert(0, None)

    return seq

def loadNoise(d, shape):
    '''
    Load gaussian random sequence with seed = 1.
    this could be done much better. Things to improve are:
    1. Not have the folder hardcoded to point at Gaussian random noise
    2. Not have the folder hardcoded to point at seed=1
    3. Have the input binary file be of different types. Here I'm assuming the binary file encodes floats wtih 4 bytes

    Output: an array of shape = ('shape', N) where N = d['endT']/d['framePeriod']/d['waitframes']
    
    shape:  touple with the shape of one frame
    d:      dictionary with experimental variables like waitframes, endT, frameperiod, etc
    '''
    import struct
    #_pdb.set_trace()

    # how many stimulus frames do we have to load?
    N = _np.ceil(d['endT']/d['framePeriod']/d['waitframes'])
    print(N)
    
    totalSamples = N*_np.array(shape).prod()

    # Given that each file holds 100K floats, how many files do I need?
    filesN = int(_np.ceil(totalSamples/1E5))

    # load all the needed files
    seq = _np.array([])
    for i in range(filesN):
        s_file = '/Users/jadz/Documents/Notebook/Matlab/RandSequences/Gaussian/Seed=1/SP_{0}-{1}K.bin'.format(i*100, (i+1)*100)
        print('loading data from: ', s_file)
        
        with open(s_file, 'rb') as fin:
            seq = _np.concatenate((seq, _np.array(struct.unpack('f'*100000, fin.read(4*100000)))))

    #_pdb.set_trace()
    seq = _np.array(seq[:totalSamples])
    seq = seq.reshape(2, -1, order='F')
    
    return seq

def flipSTA(sta):
    '''
    In GaussianNaturalStimulus, whether a cell responds to an increase in the gaussian stream generator or a decrease, corresponds not only to the polarity of the cell but also to whether the cell is on a black to white or white to black edge. 
    Here I will just assume that all cells are off, compute the max and the min of the sta and flip the sta if abs(min)>max
    
    input:
    ------
    sta:    the Spike Triggered Average in question. It has potentialy many checkers

    output:
    -------
            sta if max(sta) > abs(min(sta))
            -sta if not
    '''
    #_pdb.set_trace()
    shapeOri = sta.shape
    sta = sta.reshape(-1, sta.shape[-1])

    for ch in range(sta.shape[0]):
        if sta[ch].max() > abs(sta[ch].min()):
            sta[ch] *= -1
    
    # return sta to its original shape
    sta = sta.reshape(shapeOri)
    return sta

def getLP(sta, stim):
    '''
    convolve sta with each checker of stim

    input:
    ------
        sta:    the spike triggered average

        stim:   in this case a 2 by framesN array

    output:
    -------
        lp:     the linear prediction per checker
                in this case is a 2 by framesN - staLength array
    '''

    # reshape stim to be 2D, first dimension is checkers, 2nd dimension is time
    stim2D = stim.stim.reshape(-1, stim.stim.shape[-1])

    # reshape sta also to be 2D, first dimension is a checkers, 2nd is time
    sta2D = sta.reshape(-1, sta.shape[-1])

    # preallocate the ouptut to be the same shape as the stimulus but with a shorten time since first sta length of the stimulus can not be convolved with sta.
    lp = _np.empty((stim2D.shape[0], stim2D.shape[1]-sta2D.shape[1]+1,))

    # convolve the temporal stim in each checker with its filter
    for ch in range(lp.shape[0]):
        lp[ch] = _np.convolve(stim2D[ch], sta2D[ch], mode='valid')

    # redimension lp back to original sta shape
    lp = lp.reshape(sta.shape[:-1]+ (-1,))

    tax = _np.arange((sta.shape[-1]-1)*stim.deltaT, (sta.shape[-1]-1)*stim.deltaT + (lp.shape[-1] + 1)*stim.deltaT, stim.deltaT)
    return lp, tax

def lnWrapper(spikes, sta, stim, numbins=50):
    '''
    compute a non linearity for each checker in the stim.
    compute the linear prediction (lp) for each checker given the spikes and the sta. 
    also compute the response histogram over a similar time axis
    then call estnln(lp, r) for each checker

    output:
        nl
    '''

    from nonlinearities import estnln

    # preallocate for the nonlinearities, one per checker. Same shape as STA but instead of time they have numbins
    nl = _np.empty(sta.shape[:-1]+(numbins,))
    nl = nl.reshape(-1, numbins)

    # get the linear prediction
    lp, tax = getLP(sta, stim)

    # bin the response of the cell, exclude 1st staLength seconds
    r, _ = _np.histogram(spikes, tax)

    for ch in range(nl.shape[0]):
        #_, _, nl[ch][:], _, _ = estnln(lp[ch], r)
        _, _, n, _, _ = estnln(lp[ch], r)
        print(n.shape)
    return nl


def myPlot(sta, x, nameout=None):
    '''
    sta:    all conditions for one cell
            len(sta): # of rows
            len(sta[0]): # of cols
            for each row and col plots all checkers (ch) sta[row][col][ch] with colors from 'c' below
    '''

    rowsN = len(sta)
    colsN = len(sta[0])
    checkersN = len(sta[0][0])
    c = 'rb'
    for row in range(rowsN):
        for col in range(colsN):
            ax = _plt.subplot(rowsN, colsN, col + row*colsN+1)
            _plt.axhline(0, color='black')
            '''
            if row == 1:
                _plt.tick_params(\
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    top='off',         # ticks along the top edge are off
                    bottom='off',      # ticks along the bottom edge are off
                    labelleft='off')   # labels along the bottom edge are offax.xaxis.set_
            '''
            if row == rowsN-1:
                _plt.tick_params(\
                    axis='both',          # changes apply to the y-axis
                    which='both',      # both major and minor ticks are affected
                    left='off',      # ticks along the bottom edge are off
                    right='off',         # ticks along the top edge are off
                    labelleft='off',
                    top='off'
                    ) # labels along the bottom edge are off
            else:
                _plt.axis('off')
                '''
                _plt.tick_params(\
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    left='off',        # ticks along the left edge are off
                    right='off',         # ticks along the right edge are off
                    top='off',
                    bottom='off',
                    labelleft='off',
                    labelbottom='off')     # labels along the bottom edge are offax.xaxis.set_
                '''
            if row == 0 and col == 0:
                _plt.title('H -> T')
            elif row == 0 and col == 1:
                _plt.title('T -> H')

            for ch in range(checkersN):
                _plt.plot(x, sta[row][col][ch], c[ch])

    if nameout is not None:
        _plt.savefig(nameout, bbox_inches='tight', transparent=True)

def plotRasters(d, cells, cellnum, nameout=None, *argv, **kwargv):
    rowN = d['targetN']
    colN = 2
    _plt.figure('rasters')
    for row in range(rowN):
        for col in range(colN):
            _plt.subplot(rowN, colN, row*colN + col + 1)
            raster = ba.raster(cells[cellnum][row][col], d['fixationLength'])
            _plt.plot(raster[:,0], raster[:,1], '.k', *argv,**kwargv)

            # only show axis for the bottom plots
            if row == rowN-1:
                _plt.tick_params(\
                    axis='both',          # changes apply to the y-axis
                    which='both',      # both major and minor ticks are affected
                    left='off',      # ticks along the bottom edge are off
                    right='off',         # ticks along the top edge are off
                    labelleft='off',
                    top='off'
                    ) # labels along the bottom edge are off
            else:
                _plt.axis('off')

            # Add some title to the top plots
            if row == 0 and col == 0:
                _plt.title('H -> T')
            elif row == 0 and col == 1:
                _plt.title('T -> H')

            # scale all axis horizontally in the same way
            _plt.xlim(0, d['fixationLength'])
            #_plt.xlim(0, .3)
            
    if nameout is not None:
        _plt.savefig(nameout, bbox_inches='tight', transparent=True)

def correlate(stim, spikes, corrLength):
    '''
    correlate the stim with the spikes
    
    input:
    ------
        stim:   a stimulus object, see stimulustools
        
        spikes: a list of spike times

        corrLength: length of the correlation, in seconds

    output:
        corr:   has same time resolution as stim (stim.deltaT)

    '''
    _pdb.set_trace()
    # make stim 2D to loop over checkers
    st2d = stim.stim.reshape(-1, stim.stim.shape[-1])

    # preallocate the ndarray for corr
    corr = _np.zeros(corrLength/stim.deltaT)

    # histogram the response
    r, _ = _np.histogram(spikes, stim.tsa)

    for ch in range(st2d.shape[0]):
        onChecker = st2d[ch][:-1]
        
        for i in range(len(corr)):
            corr[i] += _np.dot(r, _np.roll(onChecker, i))

    return corr


def limitSpikes(cell, period, startT, endT):
    '''
    from all the spikes for the given cell, return only those spikes that happen in between startT and endT after being mod by period
    (this is equivalent of looking at rasters with period and accepting only spikes in between startT and endT)
    
    input:
    ------
        cell:       ndarray with spike times
        period:     float,  used to compute rx = mod(cell, period)
        startT:     flaot, the start of the time window
        endT:       float, the end of teh time window

    output:
    -------
        twspikes:   ndarray with a subset of spikes from cell
    '''

    return _np.array([spike for spike in cell if startT < _np.mod(spike, period) and _np.mod(spike, period) <= endT])

class TW:
    
    '''
    define a new class, Time Window that has in principle a few fields.
    
    startT:     float, defines the start of the time window

    endT:       float, defines the end of the time window
        
    TWcells:    nested lists of ndarrays, contains all the cells and conditions in cells but each condition only has spikes in the given TW
    
    '''
    
    def __init__(self, d, startT, endT, cells):
        '''
        inputs:
        -------

            startT:     float, defines the start of the time window

            endT:       float, defines the end of the time window
            
            cells:      nested lists of ndarrays with the spikes for all conditions and all cells 
        '''
        self.startT = startT
        self.endT = endT
        self.cells = ba.processNested(limitSpikes, 0, cells, d['fixationLength'], startT, endT)   

def getsta(*argv, **kwargv):
    #_pdb.set_trace()
    a1=ft.getsta(*argv, **kwargv)
    b1=flipSTA(a1)
    return b1

