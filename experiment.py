#!/Users/jadz/anaconda/bin/python
"""
Module to load spikes after sorting.
I'm assuming that I have run code like "exportSpikes" in the sorting computer that generates for each experiment type a text file with all spikes for all cells. Such file starts with a header (a number of cells starting with '#') and then first line after header contains all spikes for cell 1 tab delimited, second line is all spikes for cell 2 and so forth

By experiment type I refear to a line in the startT.txt file that is generated after analysing the photodiode.

"""

import numpy as _np
from collections import Counter as _Counter
from collections import defaultdict as _defaultdict
from datetime import datetime as _datetime
import json as _json
import pdb as _pdb

def loadVariables(spkFile):
    '''
    Load experimental parameters from spk file
    If a spk file starts with a header, load the experimental parameters in the header and return them in a dictionay
    Any number of lines at the beginnng of a spk file starting with '#' will be understood as comments. The '#' will be stripped and the rest of the line will be converted to a python dictionary with json

    I'm also creating a unique string to identify the retina this data is coming from based on the path to the current folder/directory. For the time being I'm assuming the path has some structure but this could be passed as a parameter. Would be nice to have a setup file in pyret for this type of parameters.
    
    input:
    ------
        spkFile:    the name of the file with the spikes

    output:
    -------
        d:  a dictionary with the parameters in the header and a string uniquely identifying this retina.
    '''
    with open(spkFile) as fid:
        header = fid.readline()[2:]

        d = _json.loads(header)

    d['s_exp'] = getExpStr()
    return d

def loadOneCell(spkFile, cell):
    '''
    Load all spikes for the given spkFile and cell.
    spkFile: a str, probably something like RF.spk, UFlicker.spk, etc
    cell:   an int representing the cell, starting from 0

    output: spks, a numpy_array
    '''
    with open(spkFile, 'r') as f_in:
        lineN = 0   # not counting comment lines
        for line in f_in:
            if line[0] == '#':
                continue
            elif lineN == cell:
                f_in.close()
                spikes = _np.fromstring(line, dtype=float, sep='\t')
                return spikes

            lineN += 1

    f_in.close()
    message = 'Trying to load cell {0} from file {1}, but the file only has {2} lines (one line per cell)'.format(cell, spkFile, lineN)

    raise ValueError(message)

def loadAllCells(spkFile):
    '''
    Load all spikes for all cells under the given experiment
    spkFile: a str, probably something like RF.spk, UFlicker.spk, etc

    output: spks, a list of numpy_array
            to obtain spikes for 'cell' do spks[cell]
    '''
    spikeList = []
    
    with open(spkFile, 'r') as f_in:
        for line in f_in:
            if line[0] == '#':
                continue
            spikeList.append(_np.fromstring(line, dtype=float, sep='\t'))


    f_in.close()
    return spikeList


def divideSpikes(spikes, blockStartT, blockEndT, blockSeq, flag):
    '''
    From spikes, generate a dictionary where keys are elements from blockSeq and values are ndarrays with all spikes in between blockStartT/EndT for that conditoin. 
    
    input:
    ------
        spikes:         ndarray like with spike times
        
        blockStartT:    ndarray like with the start time of each block
        
        blockEndT:      ndarray like with the end time of each block
        
        blockSeq:       ndarray like with 'keys' identifying each block. Blocks with the same identifier will end up together.
                        keys can be integers, strings or any other immutable object

        Flag:           Decides between different types of computations on the spikes
                        0:      Spike times are not changed at all
                        1:      Spike times are changed as if all block sequences for a given condition were continuous
                                (the time for the first instance of each block seq is 0, the second instance starts from where the 1st left off and so on)
                        2:      Spike times are changed such that EVERY block seq starts from 0

    output:
    -------
        spikesOut:      a dictionary in which spikesOut[blockSeq[i]] is a ndarray with all the spikes associated with blockSeq[i]
                        Depending on 'flag' spike times might be modified.
                        
    Usage:
    ------
        Possible use of Flag 0
            Set random seed at the beginning and have a random stimuli alternating between conditions. Both conditions draw numbers from the same random stream.
        Possible use of Flag 1
            Set the seed for as many random streams as experimental conditions and alternate the conditions many times without reseting the seed
        Possible use of Flag 2
            Reset the seed when a stimulus is repeated
    '''

    # Make a dictionary where keys are blockSeq IDs and the values are the accumulated time under such condition. This will be used if flag==2
    accumulatedTime = _Counter()

    # start an empty array where spikes will be added
    spikesOut = _defaultdict(lambda : _np.array([]))

    # add two spike to 'spikes' one prior to first blockStartT and one after last blockEndT to avoid special cases below. By adding these spikes startIndex and lastIndex are always found
    preSpk = _np.array([blockStartT[0]-1])
    postSpk = _np.array([blockEndT[-1]+1])
    spks = _np.concatenate((preSpk, spikes, postSpk))

    #_pdb.set_trace()
    for i, startT in enumerate(blockStartT):
        # only assign spikes with meaningful blockSeq. Sometimes I want to exclude spikes from the analysis for example during adapting sequences.
        if blockSeq[i] is None:
            continue
        
        # find 1st spike in spikes that is greater than startT
        startIndex = _np.where(_np.diff(_np.sign(spks-startT)))[0][0]+1
        
        # find last spike in spikes that is smaller than BlockEndT[i]
        lastIndex = _np.where(_np.diff(_np.sign(spks-blockEndT[i])))[0][0]

        # grab only the spikes corresponding to this block
        blockSpks = spks[startIndex:lastIndex+1]

        # Modify spike times in this block according to flag
        if flag==0:
            pass
        elif flag==1:
            blockSpks -= sum(accumulatedTime.values()) - accumulatedTime[blockSeq[i]]
        elif flag==2:
            blockSpks -= startT

        #_pdb.set_trace()
        # Add spike times to spikesOut
        spikesOut[blockSeq[i]] = _np.concatenate((spikesOut[blockSeq[i]], blockSpks))

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
        fid.write(_datetime.today().strftime('# %A, %B %d %Y @ %X\r'))

        # store input dictionary
        _json.dump(d, fid)
        fid.write('\r')

        # Add an empty line for human readability
        fid.write('\r')
        
        # Add old data to file
        fid.write(oldText)

def loadParameters(expName):
    '''

    Open parameters.txt and extract the dictionary values last used in processing expName
    
    for example if processing an experiment named RF, this fnction will look for a line saying ... "expName": "RF" ...
    
    input:
    ------
        expName:    the name you last used in processing the experiment.

    output:
    -------
        d:          dictionary with the parameters used in last processing expName
                    if expName is not found in parameters.txt an empty dictionary will be returned.

    '''
    
    # Since every time I process an experiment parameters are added at the beginning of the file, just read until the given 'expName' is found and then extract the parameters
    # Since some expNames might be similar don't just look for expName, but explicitly for:  '"expName": "..."' (where ... is passed parameter 'expName')
    with open('parameters.txt') as fid:
        # Scan file until a line contains "expName": "..." (where ... is passed parameter 'expName')
        # Then read that line as a dictionary and return it
        
        searchStr = '"expName": "'+expName+'"'
        for line in fid:
            if line.find(searchStr) >= 0:
                d = _json.loads(line)
                return d
    
    # if execution got to this point, expName was not found
    return {}

def getExpStr(regExp='.*([MRS/][0-9]{6}).*etina[\ ]([0-9]).*'):
    '''
    from the path to current folder where experiment is being processed, extract a string identifying the retina in questin uniquely.

    String will be of the form [SMR]YYMMDD_R# where S: salamander, R: rat, M: mouse
    '''

    from os import getcwd
    import re
    # I'm assuming 3 things:
    # 1. Date is 6 consecutive digits and there is only 1 such string
    # 2. Right in front of the date there is either '/' (default for Salamander experiments) or an M/R/S indicating species literally 
    # 3. It has a Retina or retina subfolder followed by perhaps a space and the retina number on that day (one digit). 
    reObj = re.compile(regExp)
    patterns = re.split(reObj, getcwd())
    return patterns[1].replace('/', 'S') + '_R' + patterns[2]
