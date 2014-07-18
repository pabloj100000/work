#!/Users/jadz/anaconda/bin/python
from numpy import concatenate, modf, linspace, ceil, histogram, mod, iterable, alltrue
import matplotlib.pyplot as plt
import pdb

def raster(spikes, period):
    '''
    return a 2D numpy array that has raster X/Y information for the spikes
    '''
    X, Y = modf(spikes/period)
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)

    raster = concatenate((X,Y), axis=1)
    return raster


def PSTH(spikes, period, repeats, deltaT=.05, plotFlag=0, returnFlag=0):
    '''
    Compute and return the PSTH. Depending on returnFlag, PSTH and/or x are returned.
        returnFlag: 0, returns both hist and bins
                    1, returns just hist
                    2, returns just bins
    '''
    #pdb.set_trace()
    x = linspace(0, period, ceil(period/deltaT))

    hist, bins = histogram( mod(spikes, period), bins=x )
    hist /= deltaT*repeats

    if returnFlag == 0:
        return hist, bins
    elif returnFlag == 1:
        return hist
    elif returnFlag == 2:
        return bins


def processNested(nestedList, func, *argv, recLevel=0, **kwargv):
    '''
    Spikes, PSTHs, Rasters are probably stored as bunch of nested lists until at the very end, I have an ndarray.
    Traverse through the nested lists until the ndarray and in that case process it with func.
    recLevel is used in recursion and should not be given explicitly when calling this function 

    Usage:
    L = [arange(3), arange(3,7), arange(7,10)]
    L2 = processNested(L, lambda x: x**2)
    '''
    
    #pdb.set_trace()
    print(argv)
    # First of all, work on a deep copy of nestedList.
    if recLevel == 0:
        output = nestedList.copy()
        return processNested(output, func, recLevel=1, *argv, **kwargv)

    assert recLevel>0, "recLevel is not greater than 0, it was" % recLevel
    
    # if nestedList has method shape, process it with func
    # I'm assuming that nestedList are nested object (not necessarily lists) but those objects do not have a shape method
    #pdb.set_trace()
    try:
        nestedList.shape
        #print(id(nestedList))
        #pdb.set_trace()
        nestedList = func(nestedList, *argv, **kwargv)

        #print(id(nestedList))
        return nestedList
    except:
        #pdb.set_trace()
        N = len(nestedList)
        newArgs = argFromArgs(0, argv)
        print(newArgs)
        nestedList = list(map(lambda i: processNested(nestedList[i], func, recLevel=recLevel+1, *argFromArgs(i, *argv), **kwargv), range(N)))
        
        #nestedList = list(map(lambda x: processNested(x, func, recLevel=recLevel+1, *argv, **kwargv), nestedList))
        return nestedList

def argFromArgs(i, *argv):
    '''
    argv is a set where each element is either an iterable or not.
    return another set 'setOut'
    where setOut[j] is setOut[j][i] if setOut[j] is iterable and setOut[j] if it is not an iterable
    
    In calling basicAnalysis.procesNested(nestedList, func, argv, ...)
    I might want to pass the same arg to every element in nestedList or a different value of the same argument to each element in nestedList.
    I'm envisioning a solution where before calling map
    I'm going to change argv in such a way that each argument in argv is an iterable with len set to N
    If any arg in argv is of length N nothing is being done to it.
    If any arg in argv if not iterable, then it is changed by [arg]*N
    if any arg is iterable but has more or less items than N an error is raised.
    '''
    #pdb.set_trace()
    return tuple([arg[i] if iterable(arg) else arg for arg in argv])

def plot2D(list2D, *argv, **kwargv):
    '''
    plot all list2D traces, assumes that each element in list2D is a list with the same number of elements.
    '''
    #pdb.set_trace()
    colsN = len(list2D)
    rowsN = len(list2D[0])

    # find the maximum value in list2D to adjust ylim
    from itertools import chain
    #maxV = max(chain.from_iterable(*list2D))
    maxV = max(chain(*chain(*list2D)))

    for row in range(rowsN):
        for col in range(colsN):
            plt.subplot(rowsN, colsN, col + row*colsN + 1)
            plt.plot(list2D[col][row], *argv, **kwargv)
            plt.ylim(0, maxV)


