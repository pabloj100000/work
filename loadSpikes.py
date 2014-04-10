#!/Users/jadz/anaconda/bin/python
"""
Module to load spikes after sorting.
I'm assuming that I have run code like "exportSpikes" in the sorting computer that generates for each bin file and each cell a file of the form "binFileName_c#.spk". 
I will load all those spikes into one dictionary names "spikes".
spikes.keys are the named of the binFiles and the values, are dictionaries themselves with cells as keys and arrays with spikes as values.
If there are 3 binFiles with the same base name, for example 013014a.bin, 013014b.bin and 013014c.bin then all those spikes are concatenated together under spikes['013014']

"""

from os import listdir, getcwd
from numpy import array, concatenate, fromfile

def GetSpikeFiles():	
	path = getcwd();

	spkFiles = [f for f in listdir(path) if f.endswith('spk')];
	spkFiles.sort()
	return spkFiles


def LoadSpikes():
	path = getcwd();

	# init output dictionary, starts empty
	spikes = {};

	# get all files with spikes
	spkFiles = GetSpikeFiles() 

	# loop
	for f in spkFiles:
		# for each file, break the name into its components, the expName, the cell and the timeOffset
		fname = f.rstrip('.spk').split('_c');
		expName = fname[0][0:-1]
		letter = fname[0][-1]
		timeOffset = 1650*( ord(letter)-ord('a') );
		cell = 'c'+fname[1]

		print expName, letter, cell

		# if expName is not a key in spikes, add it (associated value starts as an empty dictionary)
		if expName not in spikes:
			spikes[expName] = {}

		# if cell not in spikes yet, add another dictionary element with cell as key and an empty array as value
		if cell not in spikes[expName]:
			spikes[expName][cell] = array([])

		# append spikes from file f to the corresponding array in the dictionary. 
		# modify spike times according to timeOffset if necessary
		spikes[expName][cell] = concatenate( (spikes[expName][cell], timeOffset + fromfile(f, sep=" ")) );


	return spikes


