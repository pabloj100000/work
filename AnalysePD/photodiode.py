import meaRecording
import numpy as np

def analysePD(wildcard, whiteThresholdFactor=.8, stimDeltaT=.2):
	'''
	Load and analyse the photodiode associated with all files that match wildcard in current folder
	
	Parameters:
	-----------
	wildcard: 				string to match files, uses fnmatch, see its help if needed but * and ? are acceptable
	whiteThresholdFactor: 	a number between 0 and 1, although the code is not checking for it and code will not crash with numbers outside this range.
							white frames are defined as any crossing of threshold where threshold is whiteTrheshold*maxValueInRecording
	stimDeltaT: 			amount of time in between stimuli to force a change in white frame periods

	Output:
	-------
	startTimes:		list with times where a stimulus started
	endTimes:		list with times where a stimulus ended
	period:			distance in between white frames for a given stimulus
	'''
	
	# generate a ndarray with all the scans corresonding to a white frame and a list with the average number of scans per frame per file
	#
	whiteFrameScans, scansperframe, scanRate = loadAllWhiteFrames(wildcard, whiteThresholdFactor)
	print 'finished loading all whiteFrames from', wildcard
	
	# At this point I have finished loading PD from all files and extracting all possible white frames.
	# Remove those white frames that are of no interest, that where detected because we are probably not flipping stimulus ever frame. 
	# If for example flipping the monitor every N frames then each white frame is repeated and detected N consecutive times.

	whiteFrameScans = keepOnlyFirstWhite(whiteFrameScans, scansperframe)	# this is a list
	
	# Make lists, startT, endT with the times corresponding to each detected stimulus
	startT, endT = GetStartTEndT(whiteFrameScans, scansperframe.mean(), scanRate, stimDeltaT)
	#startT, endT, period = GetStartTEndT(whiteFrameScans, scansperframe.mean(), scanRate, stimDeltaT)

	period = getPeriod(whiteFrameScans, startT, endT, scanRate, scansperframe.mean())
	
	# change startT, endT and period to seconds.
	startT = [startT[i]/scanRate for i in range(len(startT))]
	endT 	 = [endT[i]/scanRate for i in range(len(endT))]
	period = [period[i]/scanRate for i in range(len(period))]
	
	# compute the length of each stimulus
	length = [(endT[i]-startT[i]) for i in range(len(startT))]


	'''
	# By a change in experiment I mean that the period in between white frames changes. It can also change if experiment is skipping frames.
	# Next line is somewhat complicated to understand. If needed right in a piece of paper the sequence of scans corresponding to whiteFrames
	# then we want to compute the distance between those consecutive scans but at the end, we care about the change in distance, that
	# is why we have to compare 3 consecutive whiteFrameScans. Also, independently of the computation, we always want to return the 1st element in the list
	firstWhiteIndex = [item for item in range(len(whiteFrameScans)-1) if whiteFrameScans[item+1]+whiteFrameScans[item-1]-2*whiteFrameScans[item]>scansperframe.mean()/2 or item==0]
	
	# Make a list with indexes into lastWhiteFrames per stim. This list, as firstWhiteIndex does not hold times not scans but indexes into whiteFrameScans
	lastWhiteIndex = [firstWhiteIndex[item]-1 for item in range(1,len(firstWhiteIndex))]
	lastWhiteIndex.append(len(whiteFrameScans)-1)
	
	# make the following lists:
	# Time of 1st white in a given experiment.
	startTimes = [whiteFrameScans[index]/float(scanRate) for index in firstWhiteIndex]
	# period between whiteFrames of a given stimulus
	period = [(whiteFrameScans[index+1] - whiteFrameScans[index])/float(scanRate) for index in firstWhiteIndex]
	# end of the experiment
	endTimes = [whiteFrameScans[index]/float(scanRate)+period[i] for i, index in enumerate(lastWhiteIndex)]
	'''
	#				st whites and another with their corresponding periods
	return startT, endT, length, period

def GetStartTEndT(whiteFrameScans, scansPerFrame, scanRate, stimDeltaT):
	# convert stimDeltaT (probably .2s) into a number of scans
	scansDelta = scanRate*stimDeltaT
	
	# initialize both startT, endT and period lists
	startT=[whiteFrameScans[0]]
	endT=[]
#	periodN=[0]

	# when there is a change in whiteFrames distance it can be due to skipping frames, normal termination of stimulus or premature termination of stimulus. Since the logic becomes somewhat complex I'm defining a variable to keep track of normal stimulus termination and simplify logic
	regularStimulusEnd=0
	
	# loop through every white frame, and figure out if the distance to the previous one has changed. The formula is: d1 - d2 = (w[i]-w[i-1]) - (w[i-1]-w(i-2))
	for i in range(2, len(whiteFrameScans)):
#		periodN[-1] += 1	# [-1] means the last item, whatever stimulus we are working on, just increment the number of period

		if regularStimulusEnd:
			# it doesn't matter what the distance is because this is the 1st computed distance after a regular switch between stimulus
			print 'reseting flag'
			regularStimulusEnd=0
			continue
		
		delta = whiteFrameScans[i]+whiteFrameScans[i-2]-2*whiteFrameScans[i-1]	 # this is a difference of differences, distance[i] - distance[i-1]
											# == 0 means no difference in period
											# == stimDeltaT probably means normal switch between experiments
											# Any other difference could be due to a recent switch in experiment or skipping frames
			
		if delta<scansPerFrame/2:
			# regular distance between white frames whithin a stimulus, do nothing
			continue
		elif delta-scansDelta <scansPerFrame/2:
			# the distance between white frames increased by almost exactly stimDeltaT, this is a regular stimulus end and a new one started.
			startT.append(whiteFrameScans[i])
			endT.append(2*whiteFrameScans[i-1]-whiteFrameScans[i-2])	# it finished one period after whiteFrameScans[i-1]
			regularStimulusEnd=1
		else:
			# this is a strange case, either a frame was dropped, I quit the stimulus before it was done or I forgot to add a delay in between different stimuli
			# skipping a frame
			startT.append(whiteFrameScans[i])
			endT.append(whiteFrameScans[i])

		# if it executes this is because it detected a change in PD period. We are starting a new stimulus
#		periodN.append(0)

	# append last endT
	endT.append(2*whiteFrameScans[-1]-whiteFrameScans[-2])

#	print periodN

	# compute the period (in scans not seconds because both startT and endT are in scans)
#	period = [(endT[i]-startT[i])/periodN[i] for i in range(len(startT))]

	return startT, endT#, period

def loadAllWhiteFrames(wildcard, whiteThresholdFactor):
	'''
	For all files in current directory that match wildcard, open one at a time and extract all the scans corresponding to PD
	where whiteThreshold is crossed. For each file, load PD, compute the maximum value in the PD and define as threshold anything that
	crosses that value * whiteThresholdFactor
	***********************
	VERY IMPORTANT, this will produce random white frames if there are absolutely no white frames in the PD recording
	***********************
	
	Parameters:
	-----------
	whildcard:		a string matching all files to be analyzed, see fnmatch for syntax but (* and ? are accepted)
	whiteThresholdFactor:	a number between 0 and 1, although the code is not checking for it and any number will work
	
	Output:
	-------
	whiteFrameScans:	 ndarray with the scans corresponding to the white frames
	scansperframe:		 estimated number of scans per frame
	scanRate:			from the file's header, how many samples per second are we recording.
	'''
	import os
	import fnmatch
	from numpy import append
	
	# init the ndarray to hold all white frames detected across all files.
	allWhiteScans = np.arange(0)
	accumulatedScans = 0
	
	# init a ndarray to hold the average number of scans per frame, will have one item per file loaded
	scansperframe = np.arange(0)
	
	for file in os.listdir(os.getcwd()):
		if fnmatch.fnmatch(file, wildcard):
			print 'Working on file', file
			
			header = meaRecording.getRecordingHeader(file)
			nscans = header['nscans']
			scanRate = header['scanRate']
			
			# load PD, each item corresponds to 1/scanrate seconds
			pd = loadPD(file)
			
			# Get the number of scans per frame
			scansperframe = np.append(scansperframe, getScansPerFrame(pd))
			
			# detect the maximum value in pd_array
			pd_max = np.max(pd)
			
			# define threshold that has to be crossed to define a white pd.
			whiteThreshold = pd_max*whiteThresholdFactor
			
			# detect crossings of threshold
			# all crossings are detected, if stimulus is apdating every n frames, each white frame will be detected n times
			whiteFrameScans = findThresholdCrossing(pd, whiteThreshold)	# this is ndarray
			
			# shift the scans to take into account that we are not loading the 1st file
			whiteFrameScans += accumulatedScans
			
			# update accumulatedScans
			accumulatedScans += nscans
			
			# append white frames for current file to list with all white frames
			allWhiteScans = np.append(allWhiteScans, whiteFrameScans)
	
	return allWhiteScans, scansperframe, scanRate

def findThresholdCrossing(array, threshold):
	'''
	Find the indexes and value of array for which threshold is crossed in the upwards direction
	
	Parameters
	----------
	array: ndarray with ints or floats
	
	threshold: an int or float value
	
	Output:
	-------
	crossings: a list of 'index', 'value' pairs
	
	'''
	shifted = np.concatenate([[np.min(array)],array[:-1]])<threshold
	crossings = array>=threshold
	crossings *= shifted
	
	return np.nonzero(crossings)[0]

def loadPD(filename, maxTime = 1650):
	# load the PD
	header = meaRecording.getRecordingHeader(filename)
	fileLength = header['nscans']/header['scanRate']
	pd_array = meaRecording.getChannel(0, maxTime, filename)
	
	return pd_array

def getPeriod(whiteFrameScans, startT, endT, scanRate, scansPerFrame):
	'''
	'''

	# not all scans in endT actually correspond to whiteFrames, for example when terminating a stimulus there is no white frame but the next whiteFrame is probably at stimDeltaT*scanRate scans afterwards.
	# Therefore I can't look in whiteFrameScans for the index but I can convert it to an array an look for the threshold crossing.
	whiteFrameArray = np.array(whiteFrameScans)

	# init output list
	period=[]

	for i in range(len(startT)):
		# what are the indices of the current startT and endT in whiteFrameScans?
		startIndex = whiteFrameScans.index(startT[i])
		endIndex = whiteFrameArray.searchsorted(endT[i])
		
		# extract all those white frames in between startIndex and endIndex
		#tempArray = whiteFrameArray[startIndex + 1:endIndex] - whiteFrameArray[startIndex:endIndex-1]
		
		period.append( 1.0*(endT[i]-startT[i])/(endIndex-startIndex) )

		# period is in scans
	return period

def getScansPerFrame(pd, estimatedFramePeriod=.01, scanrate=10000):
	'''
	compute the frameperiod from a PD recording.
	Algorithm computes the FFT of the PD and checks the freq of the maxima.
	If any of the first 100 freq is close to 1/estimatedFramePeriod then
	returns that value
	
	Parameters:
	-----------
	pd: ndarray with pd values, might be the output of loadPD('file.bin', 10)
	
	estimatedFramePeriod: 1/framerate in the stimulating monitor
	
	scanrate: comes from the recording's header
	header = meaRecording.getRecordingHeader('030713a.bin')
	header['scanRate']
	
	Output:
	-------
	frameperiod: returns the average frame period in seconds.
	framesamples: average number of samples at scanrate per frame (framesamples = frameperiod*scanrate
	
	'''
	import scipy
	import scipy.fftpack
	import sys

	# FFT the signal
	pdFFT = scipy.fftpack.rfft(pd)
	pdFreqs = scipy.fftpack.rfftfreq(pdFFT.size, 1./scanrate)
 
	# loop through the 100 frequencies with stronger FFT and if frameperiod is close to the estimated one, return it
	for i in range(100):
		fftmaxarg = pdFFT.argmax()
		maxargfreq = pdFreqs[fftmaxarg]

		if maxargfreq==0:
			frameperiod=0;
		else:
			frameperiod = 1./maxargfreq
		#print fftmaxarg, maxargfreq, frameperiod
		
		if abs(frameperiod-estimatedFramePeriod)<estimatedFramePeriod*.2:
			framesamples = frameperiod*scanrate
			return framesamples
		
		# current maxargfreq was not close to estimated one, remove the max and keep looping
		pdFFT[fftmaxarg]=0
		
	# if I got here it means that non of the first 100 maxima had a freq matching estimated one
	sys.exit('frameperiod failure: None of the 100 most relevant frequencies matched the corresponding to estimatedFramePeriod')

def keepOnlyFirstWhite(whiteFrameScans, scansPerFrame):
	'''
	Compute the distance (in scans) between a white frame (at position n) and the previous one; if the distance is comparable to scansPerFrame
	remove whiteFrameScan at position n.
	
	Parameters:
	-----------
	whiteFrameScans: ndarray
	output of findThresholdCrossing(pd, whiteThreshold)
	scansperframe:	 output of frameperiod(pd)
	
	Output:
	-------
	whiteFrameScans:	ndarray with the scans corresponding to white frames
	
	Comments:
	---------
	whiteFrameScans has the points in the photodiode recording at which threshold was crossed.
	whiteFrameScans is also a link to the time of such events since each point is spaced on average every 1/scanRate seconds
	'''
	# subtract from every element in whiteFrameScans the previous element. Subtract 0 from the 1st element
	dist = whiteFrameScans[:]-np.concatenate([[0], whiteFrameScans[0:-1]])
	
	whiteFrameScans = [whiteFrameScans[item] for item, distance in enumerate(dist) if distance > scansPerFrame.mean()*1.5 or item==0]
	return whiteFrameScans
