#!/Users/jadz/anaconda/bin/python
"""
Module to load spikes after sorting.
I'm assuming that I have run code like "exportSpikes2" in the sorting computer that generates a file with all bin files and cells sorted.
I will load all those spikes into one dictionary named "spikes".
spikes.keys are the named of the binFiles and the values, are dictionaries themselves with cells as keys and arrays with spikes as values.
If there are 3 binFiles with the same base name, for example 013014a.bin, 013014b.bin and 013014c.bin then all those spikes are concatenated together under spikes['013014'] after offseting the spikes by 1650s (which right now is hardcoded)

"""

from os import listdir, getcwd
from numpy import array, concatenate, fromfile

def GetSpikeFiles():
	''' return a list with all files in current folder that end in spk'''
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
		# for each file, read one line at a time. Each line corresonds to a single file/cell combination.
		# Each line is a series of tokens delimited by "\t"
		# token 0 is the binFile name without extension. Has the suffix ('a', 'b', ...)
		# token 1 is the cell # (of the form c1, c2, etc.)
		# token 2:end are spike times
		fid = open(f, 'r');
		for line in fid:
			tokens = line.split('\t')
			if len(tokens)>2:
				expName = tokens[0][0:-1];tokens # expName is token[0] but the last letter that was appended automatically during recording
				letter = tokens[0][-1]; 		# extract the last letter. Spikes have to be shifted in time according to its value.
				timeOffset = 1650.*( ord(letter)-ord('a') );
				cell = int(tokens[1][1:]);

				# remove tokens[0/1] such that now all tokens are spike times.
				#print tokens[-1
				spikeTime = [float(t) for t in tokens[2:-1]]

				#tokens.remove(tokens[0])
				#tokens.remove(tokens[1])
				print expName, letter, cell

				# if expName is not a key in spikes, add it (associated value starts as an empty dictionary)
				if expName not in spikes:
					spikes[expName] = {}

				# if cell not in spikes yet for the given expName, add it (cell as key and an empty array as value)
				if cell not in spikes[expName]:
					spikes[expName][cell] = array([])

				# append spikes from file f to the corresponding array in the dictionary. 
				# modify spike times according to timeOffset if necessary
				# tokens[0:1] are file name and cell
				# toekns[-1] is '\r'

				spikes[expName][cell] = concatenate( (spikes[expName][cell], timeOffset + array((spikeTime[2:-1]))) );

	return spikes


