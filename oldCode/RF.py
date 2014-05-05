#!/Users/jadz/anaconda/bin/python
from numpy import fromfile, ceil
from scipy.io import loadmat

def LoadStim(gaussianFlag, endT, checkersH, checkersV, deltaT):
	# load the noise
	# gaussianFlag: 0 loads binary checkers)
	# 				1 loads gaussian noise
 	# endT: 		total stimulus length in seconds
 	# checkersH: 	how many checkers in teh stimulus horizontal direction
	# checkersV: 	idem but vertically
	# deltaT: 		distance in time between stimulus frames. This is NOT the monitor frame but how long a given pattern of checkers stays in the monitor
 	# seed: 		seed used in the random generator
	
	if gaussianFlag:
		path='~/Documents/Notebook/Matlab/RandSequences/Gaussian/Seed=1/SP_1-100K.bin'
	else:
#		path='~/Documents/Notebook/Matlab/RandSequences/Binary/Seed=1_32x32x96000.bin'
		path='Seed=1_32x32x96000.bin';

	noise = fromfile(path, dtype='uint8')

	# compute how many checkers need to be deleted
	framesN = ceil(endT/deltaT)
	noise.resize((32,32,framesN))
	
#	noise.reshape((32,32))
	return noise

def GetPath(flag):
	'''
	just a short wave of loading whatever file with the random sequence of intensities that I used in mapping the RF,
	other files can also be added here. The assumption is that who ever uses this will adapt it to his/her needs.
	'''
	
	path = {
		0 : '~/Documents/Notebook/Matlab/RandSequences/Binary/Seed=1_32x32x96000.mat', # Binary white noise
		1 : '~/Documents/Notebook/Matlab/RandSequences/Binary/Seed=1_32x32x96000.bin', # Binary white noise
		-1 : '~/Documents/Notebook/Matlab/RandSequences/Gaussian/Seed=1/SP_1-100K.bin', #Gaussian white noise
	}

	return path[flag]


def LoadSequence(path):
	'''
	Load the file associated with path
	'''
	from scipy.io import loadmat
	from numpy import nan

	if path.endswith('.mat'):
		seq = loadmat(path)
	else:
		# do somehting else
		seq =  nan

	return seq
