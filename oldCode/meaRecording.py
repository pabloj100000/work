def getRecordingHeader(filename):
  '''
    read header from an Igor generated recording with MEArecording
    
    input
    -----
    filename: the experiment to load from
    
    output
    ------
    header: a dictionary with key:value pairs
    '''
  from numpy import fromfile
  f = open(filename, 'rb')
	#FBinRead /big endian /unsigned 32 bit headerSize
	#FBinRead /big endian, signed 16 bit refnum,type
	#FBinRead /b=2, signed 16 bit, version
	#FBinRead /b=2, unsigned 32 bit refnum,nscans
	#FBinRead /b=2 unsigned 32 bit refnum,numberOfChannels
  header = {}
  header['headerSize'] = fromfile(f, '>u4', 1)[0]
  header['type'] = fromfile(f, '>i2', 1)[0] # 32 bit big endian unsigned
  header['version'] = fromfile(f, '>i2', 1)[0]
  header['nscans'] = fromfile(f, '>u4', 1)[0]
  header['numberOfChannels'] = fromfile(f, '>u4', 1)[0]
  header['whichChan'] = fromfile(f, '>i2', 1)[0]
  header['scanRate'] = fromfile(f, '>f4',1)[0]
  header['blockSize'] = fromfile(f, '>u4', 1)[0]
  header['scaleMult'] = fromfile(f, '>f4',1)[0]
  header['scaleOff'] = fromfile(f, '>f4', 1)[0]
  header['dateSize'] = fromfile(f, '>i4', 1)[0]
  header['dateStr'] = fromfile(f, 'a'+str(header['dateSize']), 1)[0]
  header['timeSize'] = fromfile(f, '>i4', 1)[0]
  header['timeStr'] = fromfile(f, 'a'+str(header['timeSize']), 1)[0]
  header['userSize'] = fromfile(f, '>i4', 1)[0]
  header['userStr'] = fromfile(f, 'a'+str(header['userSize']), 1)[0]
  f.close
  return header

def getChannel(chan,length, filename):
  '''
    load a channel form an igor generated binary experimental file
    
    Inputs
    ------
    chan: channel number to be loaded
    0, photodiode
    length: amount of time to load in seconds
    
    filename: the file to load from
    
    output
    ------
    channel: 1D ndarray
  '''
  from numpy import fromfile
  from numpy import zeros
  import meaRecording as mea
  
  header = mea.getRecordingHeader(filename)
  blockSize = header['blockSize']
  outputLength = int(length*header['scanRate'])
  samplesNeeded = min(outputLength, header['nscans'])
  outputLength = samplesNeeded

  # Generate output, an ndarray of blockTime  = []
  output = zeros(outputLength)
  f = open(filename)
  block = 0
  while samplesNeeded>0:
    # get ready to read next block corresponding to channel in question
    f.seek(header['headerSize']+block*blockSize*header['numberOfChannels']*2+chan*blockSize*2)
    
    # figure out if we are going to pull down the whole block or just a fraction
    samplesAdded = min(samplesNeeded, blockSize)

    currentSamples = fromfile(f, '>f2', samplesAdded)
    output[block*blockSize:block*blockSize+len(currentSamples)] = currentSamples

    samplesNeeded -= samplesAdded
    block += 1

  f.close()
  return output


