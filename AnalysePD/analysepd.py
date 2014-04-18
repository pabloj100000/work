#!/Users/jadz/anaconda/bin/python
'''
	Call photodiode.analysePD for all binFiles.
	This is a wrapper function to call photodio.analysePD on all bin files

	Usage: analysepd
'''

import photodiode
from os import listdir
import subprocess

nominalFrameRate=100
waitframes=3
pauseT=.2

# make a list with all binFiles in current folder
binFiles = [f for f in listdir('.') if f.endswith('.bin')]

# potentially, there are binFiles with different prefixes, like stim1a.bin, stim1b.bin, stim2a.bin, stim3a.bin
# just make a list of unique prefixes (after removing the 'letter'+'.bin', 5 chars)
prefixes = set([name[:-5] for name in binFiles])


# open a file to write output
file_obj = open('startT.txt', 'w')


# Write a header in the file
# The first char is '#' and then it says {token# : length}
s = '#{0:9}\t{1:10}\t{2:10}\t{3:10}\t{4:10}\t{5:10}\t{6:10}\n'.format('binFile'.center(9),'startT'.center(10), 'endT'.center(10), 'length'.center(10), 'period'.center(10), 'frameperiod'.center(10), 'nameOut'.center(10))
file_obj.write(s)

# loop through all those prefixes and for each prefix, analyse the PD for all associated binFiles
for prefix in prefixes:
	# For the given wildchar passed as argument 1, execute photodiode.analysePD
	PD = photodiode.PD(prefix, nominalFrameRate)
	startT, endT, period, frameperiod = PD.analysePD(waitframes, pauseT)

	length = [endT[i] - startT[i] for i in range(len(startT))]

	# Print output to file
	for i in range(len(startT)):

		s = '{0:10}\t{1:10.5f}\t{2:10.5f}\t{3:10.5f}\t{4:10.5f}\t{5:10.5f}\t\n'.format(prefix, startT[i], endT[i], length[i], period[i], frameperiod)
		file_obj.write(s)


file_obj.close()

subprocess.call(['vim', 'startT.txt'])
