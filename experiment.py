#!/Users/jadz/anaconda/bin/python
"""
Module to load spikes after sorting.
I'm assuming that I have run code like "exportSpikes" in the sorting computer that generates for each experiment type a text file with all spikes for all cells.
By experiment type I refear to a line in the startT.txt file that is generated after analysing the photodiode.
Here I will load all those spikes into a list of arrays named "spikes".

"""

from numpy import fromstring, where, diff, sign, array, concatenate, ndarray
from collections import defaultdict, Counter
from datetime import datetime
import json
import pdb

def ToFloat(s):
    try:
        a=float(s)
    except:
        a=s
    return a

'''
def loadVariables(expName, fileName='startT.txt'):
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
'''
def loadVariables(expName):
    '''
    Load experimental parameters from spk file
    '''
    with open(expName) as fid:
        header = fid.readline()[2:]

        d = json.loads(header)
        return d

def loadOneCell(expName, cell):
    '''
    Load all spikes for the given expName and cell.
    expName: a str, probably something like RF.spk, UFlicker.spk, etc
    cell:   an int representing the cell, starting from 0

    output: spks, a numpy_array
    '''
    with open(expName, 'r') as f_in:
        lineN = 0   # not counting comment lines
        for line in f_in:
            if line[0] == '#':
                continue
            elif lineN == cell:
                f_in.close()
                spikes = fromstring(line, dtype=float, sep='\t')
                return spikes
                #return fromstring(line, dtype=float, sep='\t')

            lineN += 1

    f_in.close()
    message = 'Trying to load cell {0} from file {1}, but the file only has {2} lines (one line per cell)'.format(cell, expName, lineN)

    raise ValueError(message)

def loadAllCells(expName):
    '''
    Load all spikes for all cells under the given experiment
    expName: a str, probably something like RF.spk, UFlicker.spk, etc

    output: spks, a list of numpy_array
            to obtain spikes for 'cell' do spks[cell]
    '''
    spikeList = []
    
    with open(expName, 'r') as f_in:
        for line in f_in:
            if line[0] == '#':
                continue
            spikeList.append(fromstring(line, dtype=float, sep='\t'))


    f_in.close()
    return spikeList
'''
def tempFix(spikes, expName):
    vars = LoadVariables(expName)
    spikes -= vars['startT']
    vars['endT']-=vars['startT']
    vars['startT']=0

    return vars
'''
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

    # start an empty array where spikes will be added
    spikesOut = defaultdict(lambda : array([]))

    # add two spike to 'spikes' one prior to first blockStartT and one after last blockEndT to avoid special cases below. By adding these spikes startIndex and lastIndex are always found
    preSpk = array([blockStartT[0]-1])
    postSpk = array([blockEndT[-1]+1])
    spks = concatenate((preSpk, spikes, postSpk))

    #pdb.set_trace()
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

def writeParameters(d):
    '''
    Write dictionary d at the beginning of 'parameters.txt' in a human readable format along with a comment indicating processing date and time.
    '''
    
    # Open the file and read into memory the whole thing
    with open('parameters.txt', 'r') as fid:
        # read the whole file
        oldText = fid.read()
        
    # Now re-open the file again, this time for writing. Add the new processing information and then dump the old data
    with open('parameters.txt', 'w') as fid:
        # insert a greeting with the date
        fid.write(datetime.today().strftime('# %A, %B %d %Y @ %X\r'))

        # store input dictionary
        json.dump(d, fid)
        fid.write('\r')

        # Add an empty line for human readability
        fid.write('\r')
        
        # Add old data to file
        fid.write(oldText)

def loadParameters(expName):
    '''
    Open parameters.txt and extract the dictionary with values last used in processing expName
    '''
    
    # Since every time I process an experiment parameters are added at the beginning of the file, just read until the given 'expName' is found and then extract the parameters
    # Since some expNames might be similar don't just look for expName, but explicitly for:  '"expName": "..."' (where ... is passed parameter 'expName')
    with open('parameters.txt') as fid:
        # Scan file until a line contains "expName": "..." (where ... is passed parameter 'expName')
        # Then read that line as a dictionary and return it
        
        searchStr = '"expName": "'+expName+'"'
        for line in fid:
            if line.find(searchStr) >= 0:
                d = json.loads(line)
                return d
    
    # if execution got to this point, expName was not found
    return {}

