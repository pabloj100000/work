#!/Users/jadz/anaconda/bin/python
"""
Module to load spikes after sorting.
I'm assuming that I have run code like "exportSpikes" in the sorting computer that generates for each experiment type a text file with all spikes for all cells.
By experiment type I refear to a line in the startT.txt file that is generated after analysing the photodiode.
Here I will load all those spikes into a list of arrays named "spikes".

"""

from numpy import fromstring, where, diff, sign, array, concatenate, ndarray
from collections import defaultdict, Counter
import pdb

def ToFloat(s):
    try:
        a=float(s)
    except:
        a=s
    return a

def LoadVariables(expName, fileName='startT.txt'):
    # if expName has extension, remove it
    expName = expName.split('.')[0]

    with open(fileName) as f_in:
        # read 1st line with the name of each field
        header = f_in.readline()[1:]

        # if any field starts with 'w_', remove it ('w_' means wave in igor)
        header = header.replace('w_', '')

        # split header into tokens
        fields = header.split()
        
        #pdb.set_trace()
        for line in f_in:
            values = line.split()
            # compare expName with last value in 'values' but remove potential extension
            if values[-1].split('.')[0] == expName:
                vals = {fields[i]:ToFloat(values[i]) for i in range(len(fields))}
                return vals

def LoadOneCell(expName, cell):
    '''
    Load all spikes for the given expName and cell.
    expName: a str, probably something like RF.spk, UFlicker.spk, etc
    cell:   an int representing the cell, starting from 0

    output: spks, a numpy_array
    '''
    with open(expName, 'r') as f_in:
        lineN = 0
        for line in f_in:
            if lineN == cell:
                f_in.close()
                spikes = fromstring(line, dtype=float, sep='\t')
                tempFix(spikes, expName)
                return spikes
                #return fromstring(line, dtype=float, sep='\t')

            lineN += 1

    f_in.close()
    message = 'Trying to load cell {0} from file {1}, but the file only has {2} lines (one line per cell)'.format(cell, expName, lineN)

    raise ValueError(message)

def LoadAllCells(expName):
    '''
    Load all spikes for all cells under the given experiment
    expName: a str, probably something like RF.spk, UFlicker.spk, etc

    output: spks, a list of numpy_array
            to obtain spikes for 'cell' do spks[cell]
    '''
    spikeList = []
    
    with open(expName, 'r') as f_in:
        for line in f_in:
            spikeList.append(fromstring(line, dtype=float, sep='\t'))


    f_in.close()
    return spikeList

def tempFix(spikes, expName):
    vars = LoadVariables(expName)
    spikes -= vars['startT']
    vars['endT']-=vars['startT']
    vars['startT']=0

    return vars

def divideSpikes(spikes, blockStartT, blockEndT, blockSeq, flag):
    '''
    From spikes, generate a list of arrays. List element 'i' holds all spikes associated with blockSeq 'i'
    
    spikes:         array like with spike times
    blockStartT:    array like with the start time of each block
    blockEndT:      array like with the end time of each block
    blockSeq:       array like with integers identifying each block. Blocks with teh same identifier will end up together.
    Flag:           Decides between different types of computations on the spikes
                    0:      Spike times are not changed at all
                    1:      Spike times are changed as if all block sequences for a given condition were continuous (the time for the first instance of each block seq is 0)
                    2:      Spike times are changed such that EVERY block seq starts from 0

    Usage:
    Possible use of Flag 0
        Set random seed at the beginning and have a random stimuli alternating between conditions. Both conditions draw numbers from the same stream.
    Possible use of Flag 1
        Set the seed for as many random streams as experimental conditions and alternate the conditions many times without reseting the seed
    Possible use of Flag 2
        Reset the seed when a stimulus is repeated
    '''

    # Make a dictionary where keys are blockSeq IDs and the values are the accumulated time under such condition. This will be used if flag==2
    accumulatedTime = Counter()#defaultdict(lambda : array([]))

    # start an empty list where spikes will be added
    spikesOut = defaultdict(lambda : array([]))

    # add two spike to 'spikes' one prior to first blockStartT and one after last blockEndT to avoid special cases below. By adding these spikes startIndex and lastIndex are always found
    preSpk = array([blockStartT[0]-1])
    postSpk = array([blockEndT[-1]+1])
    spks = concatenate((preSpk, spikes, postSpk))

    pdb.set_trace()
    for i, startT in enumerate(blockStartT):
        # find 1st spike in spikes that is greater than startT
        startIndex = where(diff(sign(spks-startT)))[0][0]+1
        
        # find last spike in spikes that is smaller than BlockEndT[i]
        lastIndex = where(diff(sign(spks-blockEndT[i])))[0][0]

        # grab only the spikes corresponding to this block
        blockSpks = spks[startIndex:lastIndex+1]

        # Modify spike times according to flag
        if flag==0:
            pass
        elif flag==1:
            blockSpks -= sum(accumulatedTime.values()) - accumulatedTime[blockSeq[i]]
        elif flag==2:
            blockSpks -= startT

        #pdb.set_trace()
        # Add spike times to spikesOut
        spikesOut[blockSeq[i]] = concatenate((spikesOut[blockSeq[i]], blockSpks))

        # Keep track of accumulatedTime
        accumulatedTime[blockSeq[i]] += blockEndT[i] - blockStartT[i]

    return spikesOut