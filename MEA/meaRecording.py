from numpy import fromfile
from numpy import zeros

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
  # whichChan is a list of recorded channels. It has as many items as recorded channels. Each channel
  # is a 2 byte signed integer
  header['whichChan'] = []
  for i in range(header['numberOfChannels']):
	header['whichChan'].append(fromfile(f, '>i2', 1)[0])

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
  
  header = getRecordingHeader(filename)
  blockSize = header['blockSize']
  # Change length into scans or number of scans
  scansRequested = int(length*header['scanRate'])
  # Make sure that the scansRequested is not bigger than the scans available
  scansRequested = min(scansRequested, header['nscans'])
  # I'm going to loop through the file, adding scans until scansNeeded < 0
  scansNeeded = scansRequested

  # Generate output, an ndarray of blockTime  = []
  output = zeros(scansRequested)
  f = open(filename)
  block = 0
  while scansNeeded>0:
    # get ready to read next block corresponding to channel in question
    f.seek(header['headerSize']+block*blockSize*header['numberOfChannels']*2+chan*blockSize*2)
    
    # figure out if we are going to pull down the whole block or just a fraction
    scansAdded = min(scansNeeded, blockSize)

    currentSamples = fromfile(f, '>f2', scansAdded)
    output[block*blockSize:block*blockSize+len(currentSamples)] = currentSamples

    scansNeeded -= scansAdded
    block += 1

  f.close()
  return output


