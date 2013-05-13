from readHeader import readHeader

file1 = 'headerTest1.txt'
file2 = 'headerTest2.txt'
exp2 = {'float':3.14, 'list1': [1,2,3,4], 'str1': 'asdf', 'str2':'string2','var1':3}
exp1 = exp2.copy()
exp1['d']={'value1':1,'value2':'2','value3':3.0}

def test_1():
	global exp1, file1
	obj1 = readHeader(file1)
	assert obj1 == exp1

def test_2():
	global exp2, file2
	obj2 = readHeader(file2, ':')
	assert obj2 == exp2

