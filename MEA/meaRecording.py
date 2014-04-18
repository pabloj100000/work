from numpy import fromfile
from numpy import zeros

class MEA:

	def __init__(self, filename):
		
		'''
		read header from an Igor generated recording with MEArecording
		
		input
		-----
		filename: the experiment to load from
		'''

		self.file = open(filename, 'rb')
		self.filename = filename
		self.headerSize = fromfile(self.file, '>u4', 1)[0]
		self.type = fromfile(self.file, '>i2', 1)[0] # 32 bit big endian unsigned
		self.version = fromfile(self.file, '>i2', 1)[0]
		self.nscans = fromfile(self.file, '>u4', 1)[0]
		self.numberOfChannels = fromfile(self.file, '>u4', 1)[0]
		# whichChan is a list of recorded channels. It has as many items as recorded channels. Each channel
		# is a 2 byte signed integer
		self.whichChan = []
		for i in range(self.numberOfChannels):
			self.whichChan.append(fromfile(self.file, '>i2', 1)[0])

		self.scanRate = fromfile(self.file, '>f4',1)[0]
		self.blockSize = fromfile(self.file, '>u4', 1)[0]
		self.scaleMult = fromfile(self.file, '>f4',1)[0]
		self.scaleOff = fromfile(self.file, '>f4', 1)[0]
		self.dateSize = fromfile(self.file, '>i4', 1)[0]
		self.dateStr = fromfile(self.file, 'a'+str(self.dateSize), 1)[0]
		self.timeSize = fromfile(self.file, '>i4', 1)[0]
		self.timeStr = fromfile(self.file, 'a'+str(self.timeSize), 1)[0]
		self.userSize = fromfile(self.file, '>i4', 1)[0]
		self.userStr = fromfile(self.file, 'a'+str(self.userSize), 1)[0]
		self.file.close()

	def getChannel(self, chan, length):
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
		
		# if file is closed, open it
		if self.fileClosed():
			self.file=open(self.filename)

		# Change length into scans or number of scans
		scansRequested = int(length*self.scanRate)
		# Make sure that the scansRequested is not bigger than the scans available
		scansRequested = min(scansRequested, self.nscans)
		# I'm going to loop through the file, adding scans until scansNeeded < 0
		scansNeeded = scansRequested

		# Generate output, an ndarray of blockTime	= []
		output = zeros(scansRequested)
#		f = open(self.file)
		block = 0
		while scansNeeded>0:
			# get ready to read next block corresponding to channel in question
			self.file.seek(self.headerSize+block*self.blockSize*self.numberOfChannels*2+chan*self.blockSize*2)
		
			# figure out if we are going to pull down the whole block or just a fraction
			scansAdded = min(scansNeeded, self.blockSize)
	
			currentSamples = fromfile(self.file, '>f2', scansAdded)
			output[block*self.blockSize:block*self.blockSize+len(currentSamples)] = currentSamples

			scansNeeded -= scansAdded
			block += 1

		self.file.close()
		return output

	def close(self):
		self.file.close()

	def fileClosed(self):
		return self.file.closed
