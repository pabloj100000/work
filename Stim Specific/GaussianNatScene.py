#!/Users/jadz/anaconda/bin/python
import experiment as exp
import basicAnalysis as ba
import numpy as np
import pdb

def preProcess(expName, spikeFile):
	# load exp variables
	d = exp.loadVariables(spikeFile)

	# add any previously used parameters from parameters.txt
	d.update(exp.loadParameters(expName))

	# some basic calculations
	d['fixationLength'] *= d['PDperiod']
	d['saccadesN'] = int(np.round(d['endT']/d['fixationLength']))
	
	# load all cells
	cells = exp.loadAllCells(spikeFile)

	# Load the sequence of home and targets
	blockSeq = Load_HT_Seq(d['saccadesN'])

	# Get each start/end time per block
	blockStartT = [d['fixationLength']*i for i in range(d['saccadesN'])]
	blockEndT = [blockStartT[i]+d['fixationLength'] for i in range(d['saccadesN'])]

	# loop through all cells and divide spikes for each cell according to corresponding H or T		
	cells = [exp.divideSpikes(cell, blockStartT, blockEndT, blockSeq, 0) for cell in cells] 

	return d, cells

def process(expName='GNS', spikeFile='GaussianNatScene.spk'):
	d, cells = preProcess(expName, spikeFile)

	return d, cells

def Load_HT_Seq(N):
	'''
	Load the Home and Target sequence
	'''

	# Load sequence of targets (T, randomized). In version 1 of the experiment, the stimulus intercalates H in between targets starting from H
	# Final sequence is H,T0,H,T1,H, ...
	with open('/Users/jadz/Documents/Notebook/Design and Analysis/GaussianNatScene/TargetSequence.bin', 'rb') as f_in:
		T = [c-1 for c in f_in.read()]

	# I'm going to intercalate H in target sequence. Although there is only one H in this version, I want to distinguish Hs based on the previous target. 
	# There are 'tartetsN' total targets and therefore 2*targetsN identifiers in the sequence.
	# The stim H->T[i] gets assigned identifier T[i]
	# The stim T[i]->H gets assigned identifier targetsN+T[i]
	targetsN = 8
	seq = [T[i//2] if i%2==0 else T[i//2]+targetsN for i in range(N)]

	return seq
