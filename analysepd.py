#!/Users/jadz/anaconda/bin/python

import sys
import photodiode

startT, endT, period = photodiode.analysePD(str(sys.argv[1]))

file_obj = open('test.txt', 'w')

s = '#{0:9}\t{1:10}\t{2:10}\n'.format('startT'.center(9), 'endT'.center(10), 'period'.center(10))
file_obj.write(s)

for i in range(len(startT)):
	s = '{0:10.5f}\t{1:10.5f}\t{2:10.5f}\n'.format(startT[i], endT[i], period[i])
	file_obj.write(s)

file_obj.close()
