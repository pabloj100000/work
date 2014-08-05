'''
responsetools.py

Tools for getting response properties like latency, spike count, word identity, etc

Code is organized around the idea of events you want to analyze. Each event has startT and endT. It is not assumed that one event starts where the previous one ends so taht you can have non contiguous events or even overlapping events.

eventsStartT and eventsEndT should have the same number of points.
'''

import numpy as _np
import pdb as _pdb

def getLatency(spikes, eventStartT, eventEndT, noSpikeSymbol = None):
    '''
    inputs:
    -------
        spikes:         ndarray, spike times
        
        eventStartT:    iterable, event start times
        
        eventEndT:      iterable, event end times

        noSPikeSymbol:  what to return when no spike detected in event

    output:
    -------
        latency:        ndarray, latency per event. len(latency) is len(eventStartT)

    '''
    
    # init Latency to be an ndarray of the right size, filled with noSpikeSymbols
    latency = _np.array([noSpikeSymbol for i in eventStartT])

    # Loop over events, extract spikes in a given event and if any, return timing of the 1st one after subtracting event[0]
    for i, event in enumerate(zip(eventStartT, eventEndT)):
        spikesInEvent = spikes[_np.logical_and(event[0]<spikes, spikes<=event[1])]
        if len(spikesInEvent):
            latency[i] = spikesInEvent[0] - event[0]

    #return _np.array([  _np.where(   _np.logical_and(event[0]<spikes, spikes<=event[1]), spikes-event[0], noSpikeSymbol  ).any() for event in zip(eventStartT, eventEndT)])

    return latency


def getSpkCnt(spikes, eventStartT, eventEndT):
    '''
    inputs:
    -------
        spikes:         ndarray, spike times
        
        eventStartT:    iterable, event start times
        
        eventEndT:      iterable, event end times

    output:
    -------
        spkCnt:         ndarray, number of spikes in between eventStartT and eventEndT

    '''

    return _np.array([ len(_np.where( _np.logical_and(event[0]<spikes, spikes<=event[1]))[0]) for event in zip(eventStartT, eventEndT)])
