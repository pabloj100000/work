#!/Users/jadz/anaconda/bin/python
'''
	Call photodiode.analysePD for all binFiles.
'''

import photodiode
from os import listdir

# make a list with all binFiles in current folder
binFiles = [f for f in listdir('.') if f.endswith('.bin')]

# potentially, there are binFiles with different prefixes, like stim1a.bin, stim1b.bin, stim2a.bin, stim3a.bin
# just make a list of unique prefixes.
prefixes = set([name[:-5] for name in binFiles])


# opne a file to write output
file_obj = open('test.txt', 'w')


# Write a header in the file
# The first char is '#' and then it says center across next 9 chars, tab, center across 10 chars, tab, center across 10 chars
s = '#{0:9}\t{1:10}\t{2:10}\t{3:10}\n'.format('binFile'.center(9),'startT'.center(10), 'endT'.center(10), 'period'.center(10))
file_obj.write(s)

# loop through all those prefixes and for each prefix, analyse the PD for all associated binFiles
for prefix in prefixes:
	# For the given wildchar passed as argument 1, execute photodiode.analysePD
	startT, endT, period = photodiode.analysePD(prefix+'?.bin')

	# for each element in startT, endT, period, print them
	for i in range(len(startT)):

		s = '{0:10}\t{1:10.5f}\t{2:10.5f}\t{3:10.5f}\n'.format(prefix, startT[i], endT[i], period[i])
		file_obj.write(s)


file_obj.close()


