import pdb
from numpy import fromfile
from numpy import zeros
import numpy as np
import struct

class MEA:
    def __init__(self, filename):
        '''
        Read header from an Igor generated recording with MEArecording
        Input file is assumed to be big endian

        input
        -----
        filename: the experiment to load from
        '''

        self.file = open(filename, 'rb')
        self.filename = filename
        self.headerSize = fromfile(self.file, '>I', 1)[0]  # big endian, 32 bit unsigned integer
        self.type = fromfile(self.file, '>h', 1)[0]        # big endian, 16 bit signed integer
        self.version = fromfile(self.file, '>h', 1)[0]
        self.nscans = fromfile(self.file, '>I', 1)[0]
        self.numberOfChannels = fromfile(self.file, '>I', 1)[0]
        # whichChan is a list of recorded channels. It has as many items as recorded channels. Each channel
        # is a 2 byte signed integer
        self.whichChan = []
        for i in range(self.numberOfChannels):
            self.whichChan.append(fromfile(self.file, '>i2', 1)[0])
            
        self.scanRate = fromfile(self.file, '>f',1)[0]     # big endian, 32 bit floating point
        self.blockSize = fromfile(self.file, '>I', 1)[0]
        self.scaleMult = fromfile(self.file, '>f',1)[0]
        self.scaleOff = fromfile(self.file, '>f', 1)[0]
        self.dateSize = fromfile(self.file, '>i', 1)[0]    # big endian, 32 bit signed integer
        self.dateStr = fromfile(self.file, 'a'+str(self.dateSize), 1)[0]
        self.timeSize = fromfile(self.file, '>i', 1)[0]
        self.timeStr = fromfile(self.file, 'a'+str(self.timeSize), 1)[0]    # string
        self.userSize = fromfile(self.file, '>i', 1)[0]
        self.userStr = fromfile(self.file, 'a'+str(self.userSize), 1)[0]
        self.file.close()

    def writeHeader(self, f_out):
        '''
        Write header to file in binary format
        Write is written in big endian format as was the original recording. 
        At some point it would be nice to get a flag for either big or small endian
        '''

        with open(f_out, 'wb') as fid:
            # for a description of packing types see https://docs.python.org/3.1/library/struct.html#struct-alignment
           # fid.write(self.headerSize.tobytes())
            fid.write(struct.pack('>I', self.headerSize))       # i,I are 4 bytes; h,H are 2 bytes
            fid.write(struct.pack('>h', self.type))             # i,h are signed; H, I are unsigned
            fid.write(struct.pack('>h', self.version))
            fid.write(struct.pack('>I', self.nscans))
            fid.write(struct.pack('>I', self.numberOfChannels))
            # whichChan is a list of recorded channels. It has as many items as recorded channels.
            # Each channel is a 2 byte signed integer
            for i in range(self.numberOfChannels):
                fid.write(struct.pack('>h', self.whichChan[i]))

            fid.write(struct.pack('>f', self.scanRate))
            fid.write(struct.pack('>I', self.blockSize))
            fid.write(struct.pack('>f', self.scaleMult))
            fid.write(struct.pack('>f', self.scaleOff))
            fid.write(struct.pack('>i', self.dateSize))
            fid.write(struct.pack(str(self.dateSize)+'s', self.dateStr))
            fid.write(struct.pack('>i', self.timeSize))
            fid.write(struct.pack(str(self.timeSize)+'s', self.timeStr))
            fid.write(struct.pack('>i', self.userSize))
            fid.write(struct.pack(str(self.userSize)+'s', self.userStr))

    def getChannel(self, chan, length):
        '''
        load a channel from an igor generated binary experimental file in chunk format
        This is the file originally created by David. Default parameters were such that 
        output file had 20000 samples of ch[0], then 20k of ch[1], ... 20k of ch[63], and 
        then next 20k of ch[0], 20k of ch[1], etc...

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

        #pdb.set_trace()
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

            currentSamples = fromfile(self.file, '<H', scansAdded)
            # >H doesn't work
            # <H, int16, does work

            #currentSamples = fromfile(self.file, '>f2', scansAdded)
            output[block*self.blockSize:block*self.blockSize+len(currentSamples)] = currentSamples

            scansNeeded -= scansAdded
            block += 1

        self.file.close()
        return output

    def close(self):
        self.file.close()

    def fileClosed(self):
        return self.file.closed

