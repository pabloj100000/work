#!/Users/jadz/anaconda/bin/python
import numpy as _np
import matplotlib.pyplot as _plt
import pdb as _pdb

def raster(spikes, period):
    '''
    return a 2D numpy array that has raster X/Y information for the spikes
    '''
    # modf returns fractional and integral part of a number. Fraction here means fraction of a period
    X, Y = _np.modf(spikes/period)
    X = X.reshape(-1,1)*period
    Y = Y.reshape(-1,1)

    raster = _np.concatenate((X,Y), axis=1)
    return raster


def psth(spikes, period, repeats, deltaT=.05, plotFlag=0, returnFlag=0):
    '''
    Compute and return the PSTH. Depending on returnFlag, PSTH and/or x are returned.
        returnFlag: 0, returns both hist and bins
                    1, returns just hist
                    2, returns just bins
    '''
    #_pdb.set_trace()
    x = _np.linspace(0, period, _np.ceil(period/deltaT))

    hist, bins = _np.histogram( _np.mod(spikes, period), bins=x )
    hist = hist.astype(float)
    hist /= deltaT*repeats


    if returnFlag == 0:
        return hist, bins
    elif returnFlag == 1:
        return hist
    elif returnFlag == 2:
        return bins


def processNested(func, nestedArgIndex, *argv, recLevel=0, **kwargv):
    '''
    This function runs 'func' with all arguments passed on *argv and **kwargv.
    The difference between this function and just running func(*argv, **kwargv) is that one of the parameters in *argv is an arbitrarly nested list
    of lists and what is actually passed to func, is not the nested list but the embeded elements whithin the list (they can't be lists but could
    be arrays, numbers, sets, strings, etc.). All other parameters in argv will also be passed to func with the cabeat that if any of the parameters
    is also a list with the same length as the one pointed by nestedArgIndex, it is also parsed when calling func.
    See the examples and _testProcessNested below for clarification

    Spikes, PSTHs, Rasters are probably stored as bunch of nested lists until at the very end, when spike times are stored in ndarray.
    Traverse through the nested lists until the ndarray and in that case process it with func.
    recLevel is used in recursion and should not be given explicitly when calling this function 

    Usage: see _testProcessNested below
    '''
    
    #_pdb.set_trace()
    #print(argv)
    # First of all, work on a deep copy of the argument pointed by nestedArgIndex.
    if recLevel == 0:
        output = argv[nestedArgIndex].copy()
        # replace in argv the element pointed by nestedArgIndex by the copy just created 
        argv = argv[:nestedArgIndex]+(output,)+argv[nestedArgIndex+1:]
        return processNested(func, nestedArgIndex, recLevel=1, *argv, **kwargv)
    
    # if argv[nestedArgIndex] is not a list, process it with func and all corresponding parameters
    # I'm assuming that argv[nestedArgIndex] are nested lists of lists until at the very end, the last object is not a list.
    #_pdb.set_trace()
    if isinstance(argv[nestedArgIndex], list):
        #_pdb.set_trace()
        N = len(argv[nestedArgIndex])
        # I have to replace element nestedArgIndex from argv, but argv is a tuple and cannot be changed, thereofre construct a new tuple
        newTupleItem = list(map(lambda i: processNested(func, nestedArgIndex, recLevel=recLevel+1, *_argFromArgs(i, nestedArgIndex, *argv), **kwargv), range(N)))
        return newTupleItem
    else:
        return func(*argv, **kwargv)


def _argFromArgs(i, nestedArgIndex, *argv):
    '''
    argv:               a tuple
    nestedArgIndex:     an index into argv, pointing to the element with the nested list.
    i:                  an index into argv[nestedArgIndex] (in case argv[nestedArgIndex] is an interable, if i==0 argv[nestedArgIndex] might not be an interable)
    
    output:             another tuple 'tout', where:
                            if tout[j] is not iterable:
                                tout[j] is not changed
                            if len(tout[j]) != len(tout[nestedArgIndex]):
                                tout[j] is unchanged 
                            if len(tout[j]) == len(tout[nestedArgIndex]):
                                tout[j] is replaced by tout[j][i] if tout[j] is iterable and tout[j] if it is not an iterable
    
General idea
    In calling basicAnalysis.procesNested(func, nestedArgIndex, argv, ...)
    I might want to pass the same arg to every element in nestedArgIndex or a different value of the same argument to each call of func(*argv, **kwargv)
    I'm envisioning a solution where before calling map
    '''
    #Careful with strings, if an arg is passed to argv and its length matches argv[nestedArgIndex] it will be changed.
    #Not clear to me what desired behaviour is yet
    return tuple([arg[i] if isinstance(arg, list) and len(arg)==len(argv[nestedArgIndex]) else arg for arg in argv])


def nestArgument(arg, nestedLike):
    '''
    This function is meant to be used with processNested.
    In calling processNested arguments are suppossed to be non iterables or iterables with the same nesting structure as the principal argument nestedArgIndex.
    If by any chance we want to pass the same iterable to all calls of processNested(nestedArgIndex, someIterable) this will fail
    THerefore I will generate a new iterable that has the same shape as nestedLike and where each final element will be the iterable to pass to func in processNested.
    arg:            some argument, most likely an iterable
    nestedLike:     a nested structure of list, the basic structure will be copied onto output
    
    output: nestedArg
    '''

    ''' No need to write new code for this, just use:
        processNested(nestedLike, lambda x: arg)
    '''

def plot2D(list2D, x=None, filename=None, *argv, **kwargv):
    '''
    plot all list2D traces, assumes that each element in list2D is a list with the same number of elements.
    '''
    #_pdb.set_trace()
    colsN = len(list2D[0])
    rowsN = len(list2D)

    # find the maximum value in list2D to adjust ylim
    from itertools import chain
    
    #maxV = max(chain.from_iterable(*list2D))
    maxV = max(chain(*chain(*list2D)))
    minV = min(chain(*chain(*list2D)))

    _plt.figure(0)
    _plt.close()
    _plt.figure(0)
    for row in range(rowsN):
        for col in range(colsN):
            _plt.subplot(rowsN, colsN, col + row*colsN + 1)
            if x is None:
                _plt.plot(list2D[row][col], *argv, **kwargv)
            else:
                _plt.plot(x, list2D[row][col], *argv, **kwargv)

            _plt.ylim(minV, maxV)

            if row == rowsN-1:
                _plt.tick_params(\
                    axis='both',          # changes apply to the y-axis
                    which='both',      # both major and minor ticks are affected
                    left='off',      # ticks along the bottom edge are off
                    right='off',         # ticks along the top edge are off
                    labelleft='off',
                    top='off'
                    ) # labels along the bottom edge are off
            else:
                _plt.axis('off')
    
    if filename is not None:
        _plt.savefig(filename)

def _testProcessNested(case):
    arg0 = [['a','b','c'],['d','e'],[['f','g'],['h']]]
    arg1 = 2
    arg2 = [[2, 3, 4], [5], [[6], [7]]]
    arg3 = [[2, 3, 4], 5, [6, [7]]]

    if case == 0:
        # should return array([2,4,6])
        return processNested(lambda x,y:x+y, 0, _np.array([1,2,3]), _np.array([1,2,3]))
    if case == 1:
        # should return "['hola 1', 'chau 2', '2 3']"
        return processNested(lambda x, y: "{0} {1}".format(x,y), 0, ['hola', 'chau', '2'], [1,2,3])
    if case == 2:
        # should return L
        return processNested(lambda w, x, y, z: w, 0, arg0, arg1, arg2, arg3)
    if case == 3:
        # should return a nested list with the same structure as L but all values equal to p
        return processNested(lambda w, x, y, z: x, 0, arg0, arg1, arg2, arg3)
    if case == 4:
        # should return a nested list with the same structure as L but taking values from arg2, compare with case 5
        return processNested(lambda w, x, y, z: y, 0, arg0, arg1, arg2, arg3)
    if case == 5:
        # should return a nested list with the same structure as L but taking values from arg3, compare with case 4
        return processNested(lambda w, x, y, z: z, 0, arg0, arg1, arg2, arg3)
    if case == 6:
        # should return a list of tuples with the nested structure of arg0 but each tuple uses elements from all args
        # this is a nice example to see what arguments are passed to func
        return processNested(lambda w, x, y, z: (w,x,y,z), 0, arg0, arg1, arg2, arg3)
    if case == 7:
        # this should faile because we are trying to use as the nestedArgument an integer, see example 8
        return processNested(lambda w, x, y, z: (w,x,y,z), 1, arg0, arg1, arg2, arg3)
    if case == 8:
        return processNested(lambda w, x, y, z: (w,x,y,z), 1, [arg0], [arg1], [arg2], [arg3])

def _test_argFromArgs(case, *argv):
    if case == 0:
        # should return (2, 3, 'b')
        print(_argFromArgs(1,0,[1,2,3],3,['a','b','c']))
    if case == 1:
        # should return (1, 'hola', 1)
        print(_argFromArgs(0, 1, 1, ['hola', 'chau','2'], [1,2,3]))
    if case == 2:
        # should return (1, 'chau', 2)
        print(_argFromArgs(1, 1, 1, ['hola', 'chau','2'], [1,2,3]))


def _test0(v, cellsN, case):
    '''
    not funcy at all, only works with GaussianNatScene so far
    many assumptions not documented
    '''
    p = v['fixationLength']

    if case==0:
        # explicitly saying what the period is in each condition
        out = [[[p*3]*8,[p*4]*8]]*cellsN
    elif case==1:
        # explicitly saying what the period is in some conditions but other conditions inherit period from parent condition
        out = [[p*3, [p*4]*8]]*cellsN
    elif case==2:
        # explicitly saying different period in some conditions and inheriting period in other conditions
        out = [[p, list(abs(_np.random.randn(8)))]]*cellsN
    elif case==3:
        out = [[p, list(_np.arange(0.5, 4.1, .5))]]*cellsN
    
    return out

def loadSeq(filePath, shape, inputBites):
    import struct
    
    _pdb.set_trace()
    samplesN = _np.array(shape).prod()
    with open(filePath, 'rb') as fin:
        seq = _np.array(struct.unpack('f'*samplesN, fin.read(4*samplesN)))

    seq = seq.reshape(shape, order='F')


    return seq


def resampleStim(stim, N):
    '''
    resample stim in the time domain (assumed to be last dimension in stim) such that last dimension gets expanded to N time samples.
    
    each element in output array is equal to an element of stim, the rule is newP = floor(oldP * (oldN-1)/(N-1))
    
    stim:   the stimulus you want to resample
    N:      meaning depends on context
            if N > stim.shape[-1]
                then N is the new size of time domain
            else
                N is a multiplicative factor, new time domain will be stim.shape[-1]*N

    '''
    from itertools import product

    # old time samples
    oldN = stim.shape[-1]

    # newStim time samples
    if N < oldN:
        N *= oldN

    # pre allocate new stim
    newStim = _np.zeros(stim.shape[:-1] + (N,), stim.dtype)

    # generate an index array to connect points in stim to points in newStim. Next lines of code might be a bit tricky. I want to generate N samples in between 0 and oldN-1, but if I do something like linspace(0, oldN-1, N) I get only one sample at oldN-1 and this is not correct. The trick is to generate N+1 samples and make the last sample be too big and discard it
    index = _np.linspace(0, oldN, N+1)[:-1].astype(int)      # int here plays double duty. On one side I want to floor results from linspace but indexing into array requires arrays of integer or boolean type, not float
    
    #_pdb.set_trace()
    # make an iterable over all coordinates of stim (excluding time)
    arg = (range(i) for i in stim.shape[:-1])
    
    for a in product(*arg):
        originalSignal = stim[a]
        newStim[a] = originalSignal[index]
    
    return newStim

def getTimeStampArray(d):
    '''
    from experimental variables, generate the time stamp array used in pyret
    d:  a dictionary with the following keys
        framePeriod
        waitframes
        endT

    output: (ndarray) the time stamp array
    '''

    return _np.arange(0, d['endT'], d['framePeriod']*d['waitframes'])

