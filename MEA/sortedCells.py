#!/Users/jadz/anaconda/bin/python
"""
Module to load spikes after sorting.
I'm assuming that I have run code like "exportSpikes" in the sorting computer that generates for each experiment type a text file with all spikes for all cells.
By experiment type I refear to a line in the startT.txt file that is generated after analysing the photodiode.
Here I will load all those spikes into a list of arrays named "spikes".

"""

from numpy import fromstring
import pdb

def LoadExpVariables(expName, fileName='startT.txt'):
    # if expName has extension, remove it
    expName = expName.split('.')[0]

    with open(fileName) as f_in:
        # read 1st line with the name of each field
        header = f_in.readline()[1:]

        # if any field starts with 'w_', remove it ('w_' means wave in igor)
        header = header.replace('w_', '')

        # split header into tokens
        fields = header.split()
        print(fields)
        pdb.set_trace()
        for line in f_in:
            values = line.split()
            # compare expName with last value in 'values' but remove potential extension
            if values[-1].split('.')[0] == expName:
                vals = {fields[i]:values[i] for i in range(len(fields))}
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
                return fromstring(line, dtype=float, sep='\t')

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
