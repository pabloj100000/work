#!/Users/jadz/anaconda/bin/python
from numpy import concatenate, modf, linspace, ceil, histogram, mod
import pdb

def raster(spikes, period):
	'''
	return a 2D numpy array that has raster X/Y information for the spikes
	'''
	X, Y = modf(spikes/period)
	X = X.reshape(-1,1)
	Y = Y.reshape(-1,1)

	raster = concatenate((X,Y), axis=1)
	return raster

def PSTH(spikes, period, repeats, deltaT=.05):
	'''
	Compute and return the PSTH. How do I implement x axis?
	'''
	x = linspace(0, period, ceil(period/deltaT))
	return histogram( mod(spikes, period), bins=x )#/(deltaT*repeats)

