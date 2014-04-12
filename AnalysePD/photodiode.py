import meaRecording
import numpy as np

def analysePD(wildcard, whiteThreshold=.8):
  '''
    Load and analyse the photodiode associated with all files that match wildcard in current folder
    
    Parameters:
    -----------
    wildcard:            string to match files, uses fnmatch, see its help if needed but * and ? are acceptable
    whiteThreshold:      a number between 0 and 1, although the code is not checking for it and code will not crash with numbers outside this range.
                         white frames are defined as any crossing of threshold where threshold is whiteTrheshold*maxValueInRecording
    
    Output:
    -------
    startTimes:      list with times where a stimulus started
    endTimes:        list with times where a stimulus ended
    period:          distance in between white frames for a given stimulus
  '''
  import numpy as np
  
  # generate a ndarray with all the scans corresonding to a white frame and a list with the average number of scans per frame per file
  #
  whiteFrameScans, scansperframe, scanRate = loadAllWhiteFrames(wildcard, whiteThreshold)
  print 'finished loading all whiteFrames from', wildcard
  
  # At this point I have finished loading PD from all files and extracting all possible white frames.
  # Remove those white frames that are of no interest, that where detected because we are probably not flipping stimulus ever frame
  whiteFrameScans = keepOnlyFirstWhite(whiteFrameScans, scansperframe)    # this is a list
  
  # Make a list with those items that correspond to the 1st white frame for a given experiment.
  # By a change in experiment I mean that the period in between white frames changes. It can also change if experiment is skipping frames.
  # next line is somewhat complicated, to understand it. If needed right in a piece of paper the sequence of scans corresponding to whiteFrames
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
  
  #                st whites and another with their corresponding periods
  return startTimes, endTimes, period

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
    whildcard:      a string matching all files to be analyzed, see fnmatch for syntax but (* and ? are accepted)
    whiteThresholdFactor:  a number between 0 and 1, although the code is not checking for it and any number will work
    
    Output:
    -------
    whiteFrameScans:     ndarray with the scans corresponding to the white frames
    scansperframe:       estimated number of scans per frame
    scanRate:            from the file's header, how many samples per second are we recording.
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
      whiteFrameScans = findThresholdCrossing(pd, whiteThreshold)    # this is ndarray
      
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
  import numpy as np
  shifted = np.concatenate([[np.min(array)],array[:-1]])<threshold
  crossings = array>=threshold
  crossings *= shifted
  
  return np.nonzero(crossings)[0]

def loadPD(filename, maxTime = 1650):
  import meaRecording
  import numpy as np
  
  # load the PD
  header = meaRecording.getRecordingHeader(filename)
  fileLength = header['nscans']/header['scanRate']
  pd_array = meaRecording.getChannel(0, maxTime, filename)
  
  return pd_array

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
  import numpy as np
  
  # FFT the signal
  pdFFT = scipy.fftpack.rfft(pd)
  pdFreqs = scipy.fftpack.rfftfreq(pdFFT.size, 1./scanrate)
  
  for i in range(100):
    fftmaxarg = pdFFT.argmax()
    maxargfreq = pdFreqs[fftmaxarg]
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
    scansperframe:   output of frameperiod(pd)
    
    Output:
    -------
    whiteFrameScans:  ndarray with the scans corresponding to white frames
    
    Comments:
    ---------
    whiteFrameScans has the points in the photodiode recording at which threshold was crossed.
    whiteFrameScans is also a link to the time of such events since each point is spaced on average every 1/scanRate seconds
    '''
  import numpy as np
  # subtract from every element in whiteFrameScans the previous element. Subtract 0 from the 1st element
  dist = whiteFrameScans[:]-np.concatenate([[0], whiteFrameScans[0:-1]])
  
  whiteFrameScans = [whiteFrameScans[item] for item, distance in enumerate(dist) if distance > scansPerFrame.mean()*1.5 or item==0]
  return whiteFrameScans

'''
  def whiteFrameDistances(whiteFrameScans, scansperframe):
  ' ''
    compute the distance between two consecutive white frames.
    Take into account that most of the time we are not fliping stimuli every frame and therefore the white frame in the photodiode
    will be repeated. Discard this repetition and just report the distance between the 1st white frames (as if flipping every frame)
    
    Parameters:
    -----------
    whiteFrameScans: ndarray
    output of findThresholdCrossing(pd, whiteThreshold)
    scansperframe:   output of frameperiod(pd)
    
    Output:
    -------
    framesN:            list with distances between consecutive whiteframes
    whiteFrameTimes:    list with the times of the white frames in framesN
    
    Comments:
    ---------
    whiteFrameScans has the points in the photodiode recording at which threshold was cross.
    whiteFrameScans is also a link to the time of such events since each point is spaced on average 1/scanRate
    if framesN[n] != framesN[n-1] it means that a new stimuli started at time whiteFrameTimes[n] and now whiteFrame distances are every framesN[n]
    ' ''
  import numpy as np
  
  framesN=[]
  changeScans=[]
  whiteFrames = whiteFrameScans.copy()
  whiteFrames = np.round(whiteFrameScans/scansperframe)
  whiteFrames -= np.concatenate([[0], whiteFrames[0:-1]])
  print whiteFrames[0:10]
  last,accumulated = 1,0
  for item, currentDistance in enumerate(whiteFrames):
    accumulated += currentDistance
    if currentDistance!=1:
      framesN.append(accumulated)
      changeScans.append(whiteFrameScans[item])
      accumulated = 0
  
  return framesN, changeScans
'''

