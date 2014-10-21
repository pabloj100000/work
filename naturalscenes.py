'''
naturalscenes.py

A module to load and process natural scenes from Tkacik database
'''
import numpy as _np
from scipy import ndimage as _nd
from scipy.signal import decimate
from glob import glob as _glob
from itertools import product as _product
import matplotlib.pyplot as _plt
import information as _info
import pdb as _pdb
import pandas as _pd
from time import time as _time
import pickle as _pickle
import pink_noise as _pn
from os import makedirs, remove

# define simulation parameters
pixperdegree = 46       # converts degrees to pixels in image
sim_delta_t = .005           # time resolution of kernels in seconds
rw_step = .01            # in degrees
saccade_size = 6         # in degrees
sim_start_t = -.5            # in seconds
sim_end_t = 1                # in seconds
bipolar_cell_file = 'bipolar_cell_5.txt'    # file exported from Igor
                                            # work with cell 5, file 4 is quite a bit more noisy and letters become more uncorrelated
noise_corr_time = .005

g = None
images_list = None
tax = None

# define parameters for analyzing words
path_template='Data/#ms letter_length/'     #it will be initialized to Data/50ms letter_length or similar
binsN = 16
bin_rate = None

# define center pathway parameters
center_size = 1         # in degrees
center_kernel_file = '/Users/jadz/Documents/Notebook/Experiments/Simulations/Natural Images DB/New Saccade analysis/center_kernel.txt'
center_weight = 1

# define surround pathway parameters
surround_size = 2.5     # in degrees
surround_kernel_file = '/Users/jadz/Documents/Notebook/Experiments/Simulations/Natural Images DB/New Saccade analysis/surround_kernel.txt'
surround_weight = .9

# define peripheral pathway parameters, if you change the parameters defining peripheral response, run "generate_peripheral_kernel()"
periphery_size = 0      # "0" mean no spatial integration for this pathway
periphery_kernel_file = '/Users/jadz/Documents/Notebook/Experiments/Simulations/Natural Images DB/New Saccade analysis/peripheral_kernel.txt'
periphery_weight = 65   # controls the overall amplitud of peripheral kernel
periphery_exc = 1       # this controlls the relative height of the gating excitatory shift
periphery_inh = 1       # this controlls the relative height of the gating inhibitory window
gating_start_t = .05    # this controlls where gating starts
gating_end_t = .15      # this controlls where gating ends/inhibition starts
inhibition_end_t = .35  # this controlls where inhibition ends and basal starts
recovery_start_t = 0.25  # this controlls where inhibition ends and starts to recover to basal state
recovery_end_t = 0.4   # this controlls when the system returns to basal state

# define parameters for internal threshold
nl_type = 'birect'
nl_basal_threshold = 55
nl_gating_amplitud = 65
nl_units = 'linear prediction'

# define parameters for adaptation block
adaptation_type = 'memory_normalization'
adaptation_memory = 2           # in seconds
adaptation_offset = 1


def data_summary(letter_length_in_ms):
    '''
    Do everything needed to generate the figures that will go into the paper. Modify as needed
    
    '''
    global g, bin_rate, datapath
    
    #_pdb.set_trace()
    letter_length = letter_length_in_ms/1000
    datapath = 'Data/{0}ms letter_length/'.format(letter_length_in_ms)
    makedirs(datapath, exist_ok=True)
    sacc_path = 'Sacc={0}, RWS=.0{1}/'.format(saccade_size, int(rw_step*100))

    # generate a bipolar cell object.
    # It has three pathways, center, surround and periphery, each one can contribute to the membrane potential (mp).
    # Then I can add noise to the noiseless mp that is consistent with Yusuf's intracellular recordings
    # Pass that noisy mp through a nonlinearity representing [Ca] concentration
    # Pass [Ca] through an adaptive block to represent vesicle release

    bipolar = cell(letter_length_in_ms)

    print('Loading or computing g')
    g = bipolar.processAllImages()              # this will take several hours unless it is loading from a file
    
    #noisy_g = bipolar.add_mp_noise(g, 0)
    print('computing and adding noise to g')
    noise_std = bipolar.noise_model(g.std(axis=0))
    noisy_g = g + 1*bipolar.get_noise(g.shape, noise_std, noise_corr_time, save_flag=1)
    #noisy_g = g

    # Compute letters at all times under all nonlinearities
    print('Computing letters at all times under both basal and gating nonlinearities')
    basal_letters = bipolar.nl_basal.torate(noisy_g)
    gating_letters = bipolar.get_gating_letters(noisy_g)

    # make a 'mixed' version of letters that takes letters form basal, gating and inh depending on overlap between time and peripheral pathway
    #gating_start_p  = int((gating_start_t - sim_start_t)/sim_delta_t)
    #gating_end_p    = int((gating_end_t - sim_start_t)/sim_delta_t)
    #inh_end_p       = int((inhibition_end_t - sim_start_t)/sim_delta_t)
    
    #_pdb.set_trace()
    #mixed_letters   = _np.concatenate((basal_letters[:, :gating_start_p], gating_letters[:, gating_start_p:gating_end_p], inh_letters[:, gating_end_p:inh_end_p], basal_letters[:, inh_end_p:]), axis=1)

    # Downsample simulation to have time step equivalent to letter_length. It should average g/basal/mixed over a letter.
    """
    noisy_g         = decimate(noisy_g, int(letter_length/sim_delta_t), axis=1)
    basal_letters   = d/ecimate(basal_letters, int(letter_length/sim_delta_t), axis = 1)
    mixed_letters   = decimate(mixed_letters, int(letter_length/sim_delta_t), axis = 1)
    """
    #return noisy_g, basal_letters
    noisy_g          = average(noisy_g, int(letter_length/sim_delta_t), 0)
    basal_letters    = average(basal_letters, int(letter_length/sim_delta_t), 0)
    gating_letters   = average(gating_letters, int(letter_length/sim_delta_t), 0)
    preSacP = int((-.1-sim_start_t)/letter_length)
    postSacP = int((.1-sim_start_t)/letter_length)
    

    #_pdb.set_trace()
    # Compute average firing rate under basal and gating
    basal_fr = basal_letters.mean(axis=0).tofile(datapath + 'basal_fr')
    gating_fr = gating_letters.mean(axis=0).tofile(datapath + 'gating_fr')

    #_pdb.set_trace()
    noisy_g[:, preSacP].tofile(sacc_path+'g_preSac_nobinning')
    basal_letters[:, preSacP].tofile(sacc_path+'basal_letters_preSac_nobinning')
    gating_letters[:, preSacP].tofile(sacc_path+'gating_letters_preSac_nobinning')
    noisy_g[:, postSacP].tofile(sacc_path+'g_postSac_nobinning')
    basal_letters[:, postSacP].tofile(sacc_path+'basal_letters_postSac_nobinning')
    gating_letters[:, postSacP].tofile(sacc_path+'gating_letters_postSac_nobinning')

    if not _np.all(basal_letters[:,0]==gating_letters[:,0]):
        print('basal_letters[:, 0] not equal to gating_letters[:,0]')
        _pdb.set_trace()

    # bin g and the letters (information calculations will now use these binned versions)
    # if using binning_type = 1 I might want to define the bins on the gating window rather than with the whole range of times. Imagine having a saccade once in a blue moon, then defining bins based on percentiles will result in gating contributing very little to defining the bins. As a result, during gating bins, rather than being uniformly distributed, will be heavily occupied in the borders (U shape) and information will decrease.
    # digitize works on 1d array but not nd arrays. So I pass the flattened version of x and then reshape back into x's original shape at the end
    print('Digitizing linear prediction and responses')
    binning_type = 1     # 1: uses percentiles, 0: equidistant bins
    # bin g, basal and gating using percentiles defined during gating
    percentiles = list(_np.arange(0, 100.1, 100/binsN))
    bins = _np.percentile(noisy_g[:, int((.12-sim_start_t)/letter_length)], percentiles)
    binned_g        = _np.digitize(noisy_g.flatten(), bins).reshape(noisy_g.shape)
    bins = _np.percentile(gating_letters[:, int((.12-sim_start_t)/letter_length)], percentiles)
    binned_basal    = _np.digitize(basal_letters.flatten(), bins).reshape(basal_letters.shape)
    binned_gating    = _np.digitize(gating_letters.flatten(), bins).reshape(gating_letters.shape)

    if not _np.all(binned_basal[:,0] == binned_gating[:,0]):
        print('binned_basal[:, 0] not equal to binned_gating[:,0]')
        _pdb.set_trace()

    # save just a few samples of g and letters (at only -.1 and .1 secs) for displaying purposes
    binned_g[:,preSacP].tofile(datapath+'binned_g_preSac')
    binned_basal[:,preSacP].tofile(datapath+'binned_basal_preSac')
    binned_gating[:,preSacP].tofile(datapath+'binned_gating_preSac')
    binned_g[:,postSacP].tofile(datapath+'binned_g_postSac')
    binned_basal[:,postSacP].tofile(datapath+'binned_basal_postSac')
    binned_gating[:,postSacP].tofile(datapath+'binned_gating_postSac')
   
    # At this point, I want data to be tuples. I will convert them here (doing it only once rather than doing it every time I need them).
    # I'm actually converting the transverse because I want binned_g[i] to be all 'g' values at time point i
    binned_g        = tuple(map(tuple, binned_g.T))
    binned_basal    = tuple(map(tuple, binned_basal.T))
    binned_gating   = tuple(map(tuple, binned_gating.T))

    """
    basal_total_info = get_total_info_since_t0(binned_g, binned_basal, -.2)
    basal_total_info.tofile(datapath+'basal_total_info')
    mixed_total_info = get_total_info_since_t0(binned_g, binned_mixed, -.2)
    mixed_total_info.tofile(datapath+'mixed_total_info')
    """
    
    #_pdb.set_trace()
    #return binned_g, binned_basal, binned_gating
    # computing information under basal and gating using different amount of letters
    print('Compute total information with N letters under basal or where the letters are taken from binned_gating')
    for N in [1, 2, 8]:
        basal_info   = mi(binned_g[N-1:], combine_consecutive_labels(binned_basal, N))
        basal_info.tofile(datapath+'mi_{0}b_0g'.format(N))
        gating_info   = mi(binned_g[N-1:], combine_consecutive_labels(binned_gating, N))
        gating_info.tofile(datapath+'mi_0b_{0}g'.format(N))

    # compare the amount of information gained by the last letter (out of N) in several different cases:
    #   i.      both letters are under basal state
    #   ii.     letters are taken from gating response
    #_pdb.set_trace()
    for i in [2, 4, 8]:
        print('Computing information last letter carries about g in a {0}L word when all letters come form basal nl'.format(i))
        basal_cond_info = cond_mi(binned_g, combine_consecutive_labels(binned_basal, i))
        basal_cond_info.tofile(datapath+'cond_mi_{0}b_0g'.format(i))         # b_: basal and g_: gating "2b_0g_" means 2 basal and 0 gating letters went into the word
        print('Computing information last letter carries about g in a {0}L word when letters come from gating'.format(i))
        gating_cond_info = cond_mi(binned_g, combine_consecutive_labels(binned_gating, i))
        gating_cond_info.tofile(datapath+'cond_mi_0b_{0}g'.format(i))
    
    """
    # compute the amount of information the given letters convey about g. By "given letters" I mean an iter
    # compare the amount of new information the last letter carries in 2 letter words as a function of letter length
    for i in [5, 25, 125]:
        print('Computing 2L cond. information from basal nl when letter length is {0}'.format(i))
        basal_cond_info = get_cond_info(binned_g, [binned_basal]*2, i/1000)
        basal_cond_info.tofile(datapath+'cond_mi_2b_0g_{0}ms'.format(i))
    """


    """
    # I'm going to compute how redundant letters after gating are with letters during gating. Instead of writing a new method, I'm creating a new array of binned_letters that has at all times, the response at gating time and passing those letters as the conditional ones to get_cond_info
    print('Compute information that 1L carries about g, conditional on a measurement at gating time')
    gating_letter = (binned_basal[time_to_point(.12,0)],)*len(binned_basal)
    basal_cond_info = get_cond_info(binned_g, [gating_letter, binned_basal])
    """

    letters_N_list = [1,2,8]
    for N in letters_N_list:
        _integrate_information(letter_length_in_ms, N)
        #_get_information_delivery_time(letter_length_in_ms, N, .8)
        #get_info_per_spike(letter_length_in_ms, N)

    # make all plots
    plot_summary(letter_length_in_ms)


def plot_summary(letter_length_in_ms):
    global g#, bin_rate, gating_start_t, gating_end_t
    
    bipolar = cell(letter_length_in_ms)

    g = bipolar.processAllImages()

    # plots that do not depend on letter length
    fig_g = plot_g(g, 100)

    bipolar.plot_noise(g, fig_g)

    fig = bipolar.plot_noise_model()

    bipolar.plot_noise_correlation(g)
    
    plot_stats_from_TNF_fits(g)

    get_image_correaltion()

    # plot how the simulation compares to gaussian flickering and the basal and gating nl
    bipolar.plot_gaussian_simulation_and_nls(g, [], [-.1,.1], [bipolar.nl_basal, bipolar.nl_gating, bipolar.nl_inh])

    # plot one TNF_psth and the result of fitting the model ot it
    still_psth, sac_psth, tax = load_TNF_PSTHs()
    bipolar._fit_PSTH(sac_psth[1,:], 'pink', .1, 128, 100, 96, range(0, 200,10), range(-50, 150, 10), 1)
    
    # plot that DO depend on letter length
    plot_calcium_information(letter_length_in_ms)

    plot_letters_N(letter_length_in_ms,[1,8],[])
    
    plot_letters_N(letter_length_in_ms, [2,4,8],[])
    datapath = path_template.replace('#', str(letter_length_in_ms))
    _plt.gcf().savefig(datapath + 'letters_N2.pdf', transparent=True)

    plot_compare_nls(letter_length_in_ms)

    plot_binned_density(letter_length_in_ms)
    
    plot_densities(letter_length_in_ms)

    plot_compare_gating_timing(letter_length_in_ms)

    plot_compare_letter_length([25,50,75])
    
    plot_gating_vs_letter_length([5,25,50,75,100,125])

    plot_information_delivery_time(letter_length_in_ms, [1,2,4,8], .85)

    plot_integrated_information(letter_length_in_ms, [1,2,4,8])

    plot_bits_per_spike(letter_length_in_ms, [1,2,4,8])


def cartoon_summary():
    '''
    This is to show why the conditional information is needed
    '''
    stim = _fake_correlated_stim(3, 1, 1000)
    tax = _np.arange(0, len(stim)*sim_delta_t, sim_delta_t)

    mi_1L = _info.mi(stim,stim)
    cond_mi = _info.cond_mi(stim[1:], stim[1:], stim[:-1])

    _plt.plot(tax, stim)
    print(mi_1L, cond_mi)

def load_infoRatio1():
    '''
    Load arrays exported from igor with the ratio of the information (saccading/basal) for the 3 cell types (fast off, slow off and on)
    File is named w_infoRatio1.txt
    '''
    info_ratio = _np.fromfile('w_infoRatio1.txt', sep=' ').reshape(3, -1)
    info_ratio_sem = _np.fromfile('w_infoRatio1_sem.txt', sep=' ').reshape(3, -1)

    fig = plot_info_ratio(info_ratio, info_ratio_sem)

    return info_ratio, info_ratio_sem, fig

def load_UFlicker_PSTH(i):
    '''
    load all files of the form UFlicker_PSTH_'i'c_#c and concatenate them together. If file doesn't exist returns None

    Choice of file name is not the best but 1st #c refers to igor's point in UFlikcer Summary experiment wave :allCells:w_mask
        If w_mask[k] is set to 1 then I have exported all PSTHs associated with that wave and name will be UFlicker_PSTH_'k'c_#c

    Second #c is the contrast.
    
    output:
    -------
        if file exists:
            psth (1d ndarray):      one long psth with all conditions one after the other

            psth_pnts (int):        number of points in a given PSTH

        if file doesn't exist:
            None
    '''

    from os import listdir

    #_pdb.set_trace()
    files = [name for name in listdir('UFlicker PSTHs') if name.startswith('UFlicker_PSTH_c{0}_'.format(i))]

    if files == []:
        return None

    files.sort(key=lambda x: int(x.split('_')[3][:-5]))
    
    psths = [_np.fromfile('UFlicker PSTHs/'+one_file, sep='\n') for one_file in files]

    x = _np.arange(0, len(psths[0])*len(files)*sim_delta_t, sim_delta_t)
    return x, _np.concatenate(psths), len(psths[0])

def load_StableObject_PSTH(psth_pnts=None):
    '''
    load psths from file StableObject_PSTH.txt, separating saccades from still.

    if psth_pnts is given then the number of samples of each psth is adjusted to be psth_pnts

    output:
    -------
        sacc_psths (1d ndarray):     one long psth with all saccading conditions one after the other

        still_psths (1d ndarray):    one long psth with all still conditions one afte the other

        psth_pnts (int):        number of points in a given PSTH
    '''

    from os import listdir
    from scipy.signal import resample

    psths = _np.fromfile('StableObject_PSTH.txt', sep='\n').reshape(-1, order = 'F')
    lumN = 4

    still_psths = psths[:len(psths)/2]
    sacc_psths = psths[len(psths)/2:]

    x = _np.linspace(0, lumN*.5, len(still_psths))
   
    #_pdb.set_trace()
    if psth_pnts is not None:
        still_psths = resample(still_psths, lumN*psth_pnts)
        sacc_psths = resample(sacc_psths, lumN*psth_pnts)
        x = _np.linspace(0, lumN*.5, lumN*psth_pnts)


    _plt.close('StablePSTHs')
    fig, ax = _plt.subplots(num='StablePSTHs')

    colors = 'krgb'
    ax.plot(x, still_psths)
    ax.plot(x, sacc_psths, ':')

    return x, still_psths.flatten(), sacc_psths.flatten()

def load_TNF_PSTHs():
    # load TNF PSTHs that were exported from igor. PSTHs for still/saccade conditions are stored in TNF_still/saccade_PSTHs.txt.
    # Each file has psth for all cells. Each PSTH has 96 pnts spaced every .005s
    still_psth = _np.fromfile('TNF_still_PSTHs.txt', sep='\t').reshape(-1, 96)
    sacc_psth = _np.fromfile('TNF_saccade_PSTHs.txt', sep='\t').reshape(-1, 96)
    tax = _np.arange(0, 96*.005, .005)

    return still_psth, sacc_psth, tax


def _correlate_images(size):
    '''
    Load images from the DB, filter them with a disk of 'size' and autocorrelate the images (after downsampling)

    Return the average image

    size:   center size in degrees.

    '''
    global images_list
    
    #_pdb.set_trace()

    if images_list is None:
        _getImagesPath()

    for i in range(len(images_list)):
        print('correlating image {0} out of {1}'.format(i, len(images_list)))
        image = _loadImage(i)

        filtered_image = _nd.uniform_filter(image, size * pixperdegree, mode='constant')

        filtered_image -= filtered_image.mean()
        filtered_image /= filtered_image.std()

        # I don't need pixel resolution and correlate is taking a long time. Therefore downsample
        # at this point, every pixel corresponds to a cell of center 'size'
        filtered_image = filtered_image[::pixperdegree/2,::pixperdegree/2]

        if 'output' not in locals():
            output = _np.zeros_like(filtered_image)

        for i in range(1,filtered_image.shape[0]):
            for j in range(1,filtered_image.shape[1]):
                output[i,j] += _np.dot(filtered_image[i:,j:].flatten(), filtered_image[:-i,:-j].flatten())/filtered_image[i:,j:].size/len(images_list)
        
    output.tofile('Data/images_auto_corr')
    return output

def combine_consecutive_labels(labels, N):
    '''
    labels is a tuple of tuple such that labels[i][j] holds the label at time 'i' and trial 'j'
    Concatenate N consecutive labels to form a word (either in stimulus or response)
    '''

    # if N == 1 maybe I shouldn't be calling this function but I include the flag here so that I don't have to worry about it when within loops
    if N==1:
        return labels

    labels_to_pass = _np.dstack((labels[N-1:],labels[N-2:-1]))
    for i in range(2, N):
        labels_to_pass = _np.dstack((labels_to_pass, labels[(N-i-1):-i]))

    
    return tuple(map(lambda x: tuple(map(tuple, x)), labels_to_pass))

def combine_labels(*labels):
    '''
    labels is a tuple of tuples of tuples 
    labels[i][j][k]     is the ith 'label' passed [j] is the time and [k] is the cell in the simulation
    for example is calling "combine_labels(basal_letters[1:], basal_letters[:-1])" then
        labels[0] refers to basal_letters[1:]
        labels[1] refers to basal_letters[:-1]

    '''
    labels = _np.dstack(labels)
    return tuple(map(lambda x: tuple(map(tuple, x)), labels))

def cond_mi(x, y, z):
    '''
    for each time point (axis 0) of x,y,z, compute MI(x[i], y[i] | z[i])
    '''
    #_pdb.set_trace()
    time_pnts = len(x)
    info = _np.zeros(time_pnts)

    for i in range(time_pnts):
        info[i] = _info.cond_mi(x[i], y[i], z[i])

    return info


def mi(x, y):
    #_pdb.set_trace()

    # x/y are tuples of tuples, such that x[i][j] represents sample j at time point i
    time_pnts = len(x)
    info = _np.zeros(time_pnts)

    for i in range(time_pnts):
        info[i] = _info.mi(x[i], y[i])

    return info

def report_Information_stats(letter_length_in_ms):
    '''
    Grab certain arrays and compute the peak, time to peak and peak duration
    '''
    #_pdb.set_trace()

    datapath = path_template.replace('#', str(letter_length_in_ms))

    basal_1L = _np.fromfile(datapath+'mi_1b_0g')
    basal_cond = _np.fromfile(datapath+'cond_mi_2b_0g')
    gating_cond = _np.fromfile(datapath+'cond_mi_0b_2g')

    traces = [basal_1L, basal_cond, gating_cond]

    # compute saccade point for the given letter_length
    saccade_pnt = int(-sim_start_t*1000/letter_length_in_ms)

    for trace in traces:
        # compute the average in between start and saccade_pnt
        avg_info = trace[:saccade_pnt+1].mean()

        max_info = trace.max()

        # threshold where 20% of the difference bewtween avg and max is reached
        threshold = avg_info + 0.2*(max_info-avg_info)

        points_above_threshold = _np.where(trace > threshold)
        print('*'*72)
        print('Maximum info is {0:2.2f}, and it is reached at {1}ms after the saccade'.format(max_info, (trace.argmax()-saccade_pnt)*letter_length_in_ms ))
        print('start = ', (points_above_threshold[0][0]-saccade_pnt)*letter_length_in_ms)
        print('end = ', (points_above_threshold[0][-1]-saccade_pnt)*letter_length_in_ms)

    print('peak/peak ratio is: ', gating_cond.max()/basal_cond[gating_cond.argmax()])


def get_image_correaltion():
    # only run this if Figures/image_correlation.pdf does not exist
    from os import listdir

    if 'image_correlation.pdf' in listdir('Figures/'):
        # Figure already exists, don't recomput it
        return None
    
    corr = _np.fromfile('Data/images_auto_corr').reshape(-1,67)

    #_pdb.set_trace()
    dist = _np.arange(0, _np.sqrt(corr.shape[0]**2 + corr.shape[1]**2)*0.5, 0.25)
    N = _np.zeros(len(dist))
    out = _np.zeros(len(dist))

    for i in range(1,corr.shape[0]-1):
        for j in range(1,corr.shape[1]-1):
            # for i, j, convert it into a distance and then into an index into dist
            index = _np.round(_np.sqrt(i**2+j**2)*0.5/.25)

            N[index]    += 1
            out[index]  += corr[i,j]

    out = _np.divide(out, N)
            
    out[0]=1
            
    _plt.close('image_correlation')
    fig, ax = _plt.subplots(num='image_correlation')

    ax.plot(dist, out, 'o')

    xticks = range(0,10,2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)
    ax.set_xlabel(r'$Angle\, (\degree)$')

    # add vertical line at experimental center size
    ax.plot([saccade_size]*2, ax.get_ylim(), ':k')

    yticks = _np.arange(0, 1.1, .5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=10)
    ax.set_ylabel(r'$Autocorrelation$')

    ax.set_xlim(0,10)

    ax.tick_params(length=3, right='off', top='off')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.subplots_adjust(left=.3, bottom=.3, right=1,top=1)
    fig.set_size_inches(2.5,2)

    fig.savefig('Figures/image_correlation.pdf', tranparent=True)
    return out
    


def get_cond_info(binned_g, letters_list):
    '''
    Compute I(g(t) ; letters_list[-1](t) | letters_list[:-1](at previous times))

    the formula reads... Compute the mutual information between g at time t and the last letter at time t, given the previous letters measured.

    
    input:
    ------
        g:                  (2d ndarray) g[i,j] is the linear prediction for cell i at time point j

        letters_list:       Each element of the list should be a 2d ndarray the same shape as g and holds the output of passing g through a given nonlinear object followed by binning

    output:
        cond_info:          at each point in time, I(x, y | z)
    
    Implementation notes:
        for each point along the time axis, extract the last value of g (x), the last letter (y) and the previous letters (z). Then feed all that into _info.cond_mi(x, y, z)

    '''
    #_pdb.set_trace()

    lettersN = len(letters_list)

    cond_info = _np.zeros(len(binned_g))

    #if lettersN > 2:
    #    raise ValueError("naturalscenes.get_cond_info:\n I'm assuming in the implementation that there are exactly 2 letters.\n Expand implementation as needed")

    # I can only compute the cond_info if last letter is such that the 1st letter is in the simulation. That means that I can't compute the cond_info for the first lettersN-1 points

    for p in range(lettersN-1, len(binned_g)):
        x = binned_g[p]

        y = letters_list[-1][p]

        z = ()
        for i in range(lettersN-1):
            # when lettersN is 2,   i=0 and letters are taken from letters_list[0][p-letters_delta_p]
            # when lettersN is 3,   i=0-> letters_list[0][p-2*letters_delta_p]
            #                       i=1-> letters_list[1][p-1*letters_delta_p]
            z += (letters_list[i][p-(lettersN-i-1)],)
        
        #newZ = tuple(zip(z))
        cond_info[p] = _info.cond_mi(x, y, newZ)

    return cond_info


def get_info(binned_g, letters_list):
    '''
    Compute I(g(t) ; letters_list[:](t and previous times))

    the formula reads... Compute the mutual information between g at time t and all letters (spaced by letter_length) ending on time t.

    
    input:
    ------
        g:                  tuple of tuples, now g[i] means all g values at time point i and g[i][j] is linear prediction for cell j, time point i

        letters_list:       Each element of the list should be a tuple of tuple as 'g', holding the output of passing g through a given nonlinear object followed by binning

    output:
        info:               at each point in time, I(x, y)
    
    Implementation notes:
        for each point along the time axis, extract the last value of g (x), the set of N letters ending on time t (y). Then feed all that into _info.mi(x, y)

    '''
    #_pdb.set_trace()

    lettersN = len(letters_list)

    info = _np.zeros(len(binned_g))

    # I can only compute the info if time is such that I can extract (lettersN-1) prior to current time. That means that for the first (lettersN-1) points I can't compute the information.
    for p in range(lettersN-1, len(binned_g)):
        x = binned_g[p]

        y = ()
        for i in range(lettersN):
            # When there is only 1 letter in letters_list, N-i-1 = 0 and 'y' is taken at point 'p' as is 'x'. With two letters, first one is taken at 'p-letters_delta_p' and second one is taken at 'p'
            y += (letters_list[i][p-(lettersN-i-1)],)
        
        newY = tuple(zip(y))#_info.combine_labels(*y)
        info[p] = _info.mi(x, newY)

    return info

def average(array_in, pnts, flag):
    '''
    return a version of array that has been resampled down by pnts along the last dimension (I'm thinking that last dimension in array_in represents time)
    
    input:
    ------
        flag:       allows for different computations to be performed when array_in does not have an integer number of pnts.
                    0, raise an error
                    1, throw away extra points at the end
                    2. throw away extra points at the beginning
                    3, average fewer points in the 1st bin
                    4, average fewer points in the last bin
            
    Implementation Notes: I will reshape array_in into a 2d array such that shape[1] = pnts. Then I'll compute the mean along axis = 1. Then reshape back into something with the same number in the 1st dimensions of shape

    '''

    extra_points = array_in.size%pnts
    if extra_points:
        if flag == 0:
            raise ValueError('naturalscenes.average: array_in.size is not an integer number of pnts')
        elif flag == 1:
            array_in = array_in[:-extra_points]
        elif flag == 2:
            array_in = array_in[extra_points:]
        elif flag == 3:
            first_bin = array_in[:extra_points].mean()
            array_in = _np.concatenate((_np.ones(pnts-extra_points)*first_bin, array_in.flatten()), axis=0)
        elif flag == 4:
            last_bin = array_in[-extra_points:].mean()
            array_in = _np.concatenate((array_in.flatten(), _np.ones(pnts-extra_points)*last_bin), axis=0)

    shape_out = array_in.shape[:-1] + (-1,)
    #array_out = array_in.reshape(-1, pnts).mean(axis=1).reshape(shape_out)
    return array_in.reshape(-1, pnts).mean(axis=1).reshape(shape_out)

    """
    #_pdb.set_trace()
    # make an array of length pnts with 1/pnts
    weights = _np.ones(pnts)/pnts

    # Convolve a flatten version of array with weights
    corr = _np.correlate(array_in.flatten(), weights, mode='full')

    # now corr can't be reshaped because it has (pnts-1) extra points
    avgArray = corr[pnts-1:].reshape(array_in.shape)

    # now all points at the end avgArray are wrong because they are mixing different rows of array
    avgArray[:, array_in.shape[1]-pnts+1:]=0

    # Extract a sub array from avgArray with points along axis=1 spaced by pnts
    outArray = avgArray[:, range(0, array_in.shape[1], pnts)]

    return outArray
    """

def _getImagesPath(path=None):
    if path is None:
        path = '/Users/jadz/Documents/Notebook/Matlab/Natural Images DB/RawData/*/*LUM.mat'
        
    global images_list
    images_list = _glob(path)

def _loadImage(imNumber):
    '''
    Load an image from the database. Image undergoes light adaptation. THe mean of the image is forced to be 127.

    inputs:
    -------
        imNumber:   integer, specifying which element from images_list to load

    output:
        image:      ndarray with the image
    '''
    from scipy import io
    global images_list

    if images_list is None:
        _getImagesPath()

    # load matlab array 
    image = io.loadmat(images_list[imNumber])['LUM_Image']

    # perform light adaptation
    image *= 127/image.mean()

    return image


def _getEyeSeq(filter_length):
    '''
    Generate a sequence of eye movements in both x and y directions
    The sequence is a 2D ndarray compossed of steps. 
    seq[0][p] is the step in the x direction at point p
    seq[1][p] is the step in the y direction at point p

    seq starts at time sim_start_t - ( len(filter_instance.center_kernel) - 1 ) * sim_delta_t and ends at time sim_end_t
    in this way, when convolving with mode='valid' the output will have samples spanning sim_start_t and sim_end_t

    intput:
    -------
        filter_length:        number of points of the filter that will be used in convolving the time sequence. filter_length -1 points are needed in front of the eye movement sequence such that the convolution with the filter will have the right number of points when using 'valid'

    output:
    -------
        seq:    2D ndarray with steps in pixels
    '''

    stepsN = int((sim_end_t-sim_start_t)/sim_delta_t + filter_length - 1)

    # generate the FEM part of the sequence
    seq = _np.random.randn(2, stepsN)
    seq *= pixperdegree*rw_step

    # add saccade in both x and y for the time being. The distribution of LP I'm getting is skewed to the right as if most images were transitioning from light to dark patches.
# I think this might be due to the fact that I'm always saccading in the same direction (may be from sky to dirt). I will randomize here the direction of the saccade but keeping both fixational points the same.
    saccadePnt = int(filter_length - 1 - sim_start_t/sim_delta_t)

    # since I'm making saccade in both x and y, amplitud of saccade is sqrt(2)*saccade_size*pixperdegree, in order to have it be of the required size I have to divide by sqrt(2)
    # If making saccades in along only x and/or y the sqrt(2) shouldn't be there and for a general saccade with angle alpha with respect ot the x axis, x
    angle_with_x_axis = _np.random.rand()*2*_np.pi
    #_pdb.set_trace()
    jump_x = saccade_size * pixperdegree * _np.cos(angle_with_x_axis)
    jump_y = saccade_size * pixperdegree * _np.sin(angle_with_x_axis)
    if jump_x > 0:
        # saccade in the usual way
        seq[0,saccadePnt] += jump_x
    else:
        seq[0,0] += jump_x
        seq[0,saccadePnt] -= jump_x
    if jump_y > 0:
        seq[1,saccadePnt] += jump_y
    else:
        # saccade backwards
        seq[1,0] += jump_y
        seq[1,saccadePnt] -= jump_y

    # change from steps to actual positions
    seq = seq.cumsum(1)

    return seq.astype('int16')


def _get_jitter_velocity():
    global saccade_size

    #_pdb.set_trace()
    # remove saccade from eye sequence
    saccade_size_ori = saccade_size
    saccade_size = 0

    # make sequence 1D
    eye_seq = _getEyeSeq(117)
    eye_seq[1,:] += eye_seq[0,-1]-eye_seq[1,0]
    eye_seq = eye_seq.reshape(-1,)

    # convert from pixels to degrees
    eye_seq = eye_seq.astype(float)
    eye_seq /= pixperdegree

    # smooth position
    N = 5
    smoothing_filter = _np.ones(N)*1/N
    smoothed = _np.correlate(eye_seq, smoothing_filter)

    # change from position to velocity
    vel = _np.diff(smoothed)

    # and from vel to speed
    speed = abs(vel)
    
    # compute average speed
    avg_speed = speed.mean()/(sim_delta_t*len(speed))


    saccade_size = saccade_size_ori
    
    print('Average speed is: ', avg_speed, 'degrees per second')
    return eye_seq, smoothed, vel, speed, avg_speed

def _get_ploting_TAX(letter_length_in_s):
    tax = _np.arange(sim_start_t, sim_end_t, letter_length_in_s)

    return tax

def _get_simulation_TAX():
    global tax
    if tax is None:
        tax = _np.arange(sim_start_t, sim_end_t, sim_delta_t)

    return tax

# plots go here
def plot_g(g, num):
    '''
    make a plot with 'num' random cells
    '''
    #_pdb.set_trace()
    tax = _get_simulation_TAX()
    
    _plt.close('g')
    fig, ax = _plt.subplots(num='g')
    
    # add 'num' traces to plot. Traces are chosen randomly from 1st dimension of g
    traces = []
    for i in range(num):
        index = _np.random.randint(0, g.shape[0])
        traces.append(ax.plot(tax, g[index,:]))

    # add dash line at saccade
    traces.append(ax.plot((0,0), ax.get_ylim(), 'k:'))

    # add labels
    ax.set_xlabel(r'$Time\, (s)$', fontsize=10)
    #ax.set_ylabel('Linear Prediction (AU)')
    ax.set_ylabel(r'$Filtered\, stimulus, (AU)$', fontsize=10, labelpad=0)
    

    xticks = _np.arange(-.2, .8, .4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)
    #ax.set_yticks((-5, 0, 5))
    ax.set_xlim(-.2, .8)
    ylim = ax.get_ylim()
    maxY = max(-ylim[0], ylim[1])

    ax.set_yticks([-maxY*.8, maxY*.8])
    ax.set_yticklabels([-1,1], fontsize=10)
    #ax.yaxis.set_visible(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.tick_params(length=3, right='off', top='off')

    fig.subplots_adjust(left=.25, bottom=.35, top=1, right=1)
    fig.set_size_inches(2, 1.5)
    fig.savefig('Figures/g.pdf', transparent=True)

    return fig


def plot_calcium_information(letter_length):
    '''
    Load and plot mutual information between binned g and different words (either 1 or 2 L and either basal or gating nl)
    '''
    #_pdb.set_trace()

    datapath = path_template.replace('#', str(letter_length))
    tax = _get_ploting_TAX(letter_length/1000)
    basal_info_1L = _np.fromfile(datapath+'mi_1b_0g')
    basal_info_2L = _np.fromfile(datapath+'mi_2b_0g')
    basal_info_8L = _np.fromfile(datapath+'mi_8b_0g')
    gating_info_1L = _np.fromfile(datapath+'mi_0b_1g')
    gating_info_2L = _np.fromfile(datapath+'mi_0b_2g')
    gating_info_8L = _np.fromfile(datapath+'mi_0b_8g')
    
    #_pdb.set_trace()
    _plt.close('calcium_information')
    fig, ax = _plt.subplots(num='calcium_information')

    ax.plot(tax, gating_info_1L, 'b', lw=2, label="gating")
    #ax.plot(tax, gating_info_2L, ':b', lw=2, label="0b 2g")
    #ax.plot(tax, gating_info_10L, ':b', lw=2, label="0b 10g")
    ax.plot(tax, basal_info_1L, 'r', lw=2, label="basal")
    #ax.plot(tax, basal_info_2L, ':r', lw=2, label="2b 0g")
    #ax.plot(tax, basal_info_10L, ':r', lw=2, label="2b 10g")

    ax.plot([0,0], (0, ax.get_ylim()[1]), ':k', label='_nolegend_')

    ax.legend(fontsize=10, handlelength=1, frameon=False, loc='lower center')
    xticks=_np.arange(-.2,.8,.4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)
    yticks=range(0,3,1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=10)
    ax.set_xlim(-.2, .8)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.tick_params(length=3, top='off', right='off')
    ax.set_xlabel(r'$Time\, (s)$')
    ax.set_ylabel(r'$Information\, (Bits)\,\,$')


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.subplots_adjust(bottom=.20, left=.2,right=1, top=1)
    fig.set_size_inches(2.5,2)
    fig.savefig(datapath + 'calcium_information.pdf', transparent=True)
    
    return fig

def plot_information():
    '''
    Load and plot mutual information between binned g and 1 letter words with basal and gating nl
    Also plot conditional information between last letter and g given the previous letter.
   
    All gating plots are in blue, basal in red, 1 letter info are lines and cond info are dotted lines
    '''
    tax = _get_ploting_TAX(letter_length/1000)
    basal_info = _np.fromfile(datapath+'mi_1b_0g_50ms')
    basal_cond_info = _np.fromfile(datapath+'cond_mi_2b_0g_50ms')
    gating_info = _np.fromfile(datapath+'mi_0b_1g_50ms')
    gating_cond_info = _np.fromfile(datapath+'cond_mi_0b_2g_50ms')
    
    _plt.close('information')
    fig, ax = _plt.subplots(num='information')

    ax.plot(tax, gating_info, 'b', lw=2)
    ax.plot(tax, gating_cond_info, ':b', lw=2)
    ax.plot(tax, basal_info, 'r', lw=2)
    ax.plot(tax, basal_cond_info, ':r', lw=2)

    ax.plot([0,0], (0, ax.get_ylim()[1]), ':k', label='_nolegend_')

    xticks=_np.arange(-.2,.8,.4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)
    yticks=range(0,3,1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=10)
    ax.set_xlim(-.2, .8)
    ax.set_ylim(0, ax.get_ylim()[1]+1)
    ax.set_xlabel(r'$Time\n (s)$')
    ax.set_ylabel(r'$Information\, (Bits)\,\,$')

    ax.tick_params(length=3, top='off', right='off')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.subplots_adjust(bottom=.20, left=.2,right=.95, top=.95)
    fig.set_size_inches(2.5,2)
    
    ax.text(-.15, 3.6, r'$gating$', color='blue', size=10)
    ax.text(-.15, 2.9, r'$basal$', color='red', size=10)
    ax.plot((-.5,- .6), (.1, .1), 'k', label = r'$I(g;L_1)$')
    ax.plot((-.5,-.6), (.2, .2), ':k', label = r'$I(g;L_1\mid L_0)$')
    ax.legend(loc = 'upper center', bbox_to_anchor=(.75, 1.05), frameon=False, fontsize=10, handlelength=2)

    fig.savefig('Figures/calcium_information.pdf', transparent=True)
    return fig

def plot_firing_rate_hist(firingRate):
    '''
    firingRate is a 2D ndarray such that firingRate[:,i] is the firing rate in a given condition.
    Histogram all firing rates in 'firingRate', to get an idea of the STD and see which one carries more entropy
    '''

    _plt.close('firingRate_std')
    fig, ax = subplots(num='firingRate_std')
    
    traces = []
    for i in range(firingRate.shape[1]):
        std = firingRate[:,i].std()
        traces.append(_plt.hist(firingRate[:, i], label=r'$\sigma={0}$'.format(std)))

    _plt.legend()
    return fig, ax, traces

def _plot_word_cond_info(tax, nogating_info, nogating_rate, gating_info, gating_rate):
    _plt.close('gating_vs_FEM_Word_cond_info')
    fig, ax = _plt.subplots(num='gating_vs_FEM_Word_cond_info')
    traces = []
    traces.append(ax.plot(tax, nogating_info, 'b', lw=2, label=r'$basal$'))
    traces.append(ax.plot(tax, gating_info, 'r', lw=2, label=r'$gating$'))
    traces.append(ax.plot([0,0], ax.get_ylim(), 'k:', label='_nolabel_'))

    ax.legend(loc='center right', fontsize=10, bbox_to_anchor=(1,.71), frameon=False, handlelength=.5, handletextpad=.1)

    ax.set_xlim(-.1, .4)
    ax.set_xticks((-.1, 0, .1, .2, .3))
    ax.set_xticklabels((-.1,"", .1,"",.3), fontsize=10)
    ax.set_ylim(0, .4)
    ax.set_yticks((0, .25))
    ax.set_yticklabels((0, .25), fontsize=10)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlabel(r'$Time\n (s)$',fontsize=10)
    ax.set_ylabel(r'$Information\, (Bits)$',fontsize=10)
    ax.yaxis.set_label_coords(-.25, .40)
    fig.subplots_adjust(bottom=.3, left=.25, right=1, top=1)

    fig.set_size_inches(2, 1.5)
    fig.savefig('Figures/gating_vs_FEM_Word_cond_info.pdf', transparent=True)

    return fig, ax, traces

    _plt.close('word_rate')
    fig = _plt.figure('word_rate')
    ax = fig.add_subplot(1,1,1)
    ax.plot(nogating_tax, nogating_rate, 'b', lw=2, label=r'$basal$')
    ax.plot(gating_tax, gating_rate, 'r', lw=2, label=r'$gating$')
    ax.legend()

def sigmoids_plot(name, tax, data, sigmoids):
    _plt.close(name)
    fig, ax = _plt.subplots(num=name)

    traces = []
    for i, sig in enumerate(sigmoids):
        traces.append(_plt.plot(tax, data[:,i], label=r'$T={0}, \sigma={1}$'.format(sig[2], sig[3])))

    _plt.legend()

    return fig, ax, traces

def explore_all_nl(g, time, nls):
    '''
    plot all nongating and gating nonlinearities simultaneously on top of the distribution of linear prediction values at the given times

    inptus:
    -------
        times:      list of times

        nls:        list of nonlinear_block objects
    '''
    #_pdb.set_trace()
    
    # close the plot if it already exists
    _plt.close('all_nls')
    fig, ax1 = _plt.subplots(num = 'all_nls')
    ax2 = ax1.twinx()

    colors = 'br'   # blue for no gating, red for gating
    bins=50
    point = time_to_point(time,0)
    data_to_hist = g[:,point]
    
    # normalize by mean and contrast
    data_to_hist -= data_to_hist.mean()
    data_to_hist /= data_to_hist.std()

    label = r'$t ={0: G}ms$'.format(int(1000*time))
    hist, bins, patches = ax1.hist(data_to_hist, bins=bins, normed=True, color='k', histtype='bar', label=label)

    #_pdb.set_trace()
    # plot nl in the same bins
    for nl in nls:
        ax2.plot(bins, nl.torate(bins)/nl.gating_nl.max_fr, colors[0], label=r'$no gating$', lw=1, alpha=.5)
        ax2.plot(bins, nl.gating_rate(bins)/nl.gating_nl.max_fr, colors[1], label=r'$gating$', lw=1, alpha=.5)
    
    #arange axis
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    ax1.set_xlabel('Filtered stimulus',fontsize=10)
    ax1.set_ylabel('',fontsize=10)
    ax1.yaxis.set_label_coords(-.25, .40)
    fig.subplots_adjust(bottom=.25, left=.10, right=.9, top=.95)

    #fig.set_size_inches(2, 1.5)
    fig.savefig('Figures/gating_vs_FEM_Word_cond_info.pdf', transparent=True)

    ax1.set_xticks((-1, 0, 1))
    ax1.set_xticklabels((-1, 0, 1), fontsize=10)
    #ax.set_xlim(-1,1)
    ax1.set_yticks((0, ax1.get_ylim()[1]))
    ax1.set_yticklabels((0,ax1.get_ylim()[1]), fontsize=10)

    ax1.text(0.2, 4, r'$t=-100ms$',color='b')
    ax1.text(0.2, 3, r'$t= 100ms$', color='r')

    ax2.set_ylim(0,ax2.get_ylim()[1])
    #ax.legend(bbox_to_anchor=(1.4,.75), fontsize=10, handlelength=.5, frameon=False)
    fig.savefig('Figures/LP_and_sigmoids.pdf', transparent=True)
    
    return fig, ax1
    
def explore_one_cell_nl(g, time, nls, bin_rate=None):
    '''
    plot the distribution of values of g at time t0 and all nonlinear objects in nls

    nls:         iterable of nonlinear_block objects

    bin_rate:   if given, rate is discretized

    '''
    _pdb.set_trace()
    # close the plot if it already exists
    _plt.close('LP_and_sigmoid')
    fig, ax1 = _plt.subplots(num = 'LP_and_sigmoid')
    ax2 = ax1.twinx()

    colors = 'rb'   # blue for no gating, red for gating
    bins=500
    point = time_to_point(time,0)
    data_to_hist = g[:,point].T
    
    # normalize by mean and contrast
    #data_to_hist -= data_to_hist.mean()
    #data_to_hist /= data_to_hist.std()
    
    label = r'$t ={0: G}ms$'.format(int(1000*time))
    #hist, bins, patches = ax1.hist(data_to_hist, bins=bins, normed=True, histtype='step', color='k', histtype='bar', label=label)
    hist, bins, patches = ax1.hist(data_to_hist, bins=bins, histtype='step', normed=True, color='k', label=label)

    # plot nl in the same range
    bins = _np.arange(bins[0], bins[-1], (bins[-1]-bins[0])/1000)
    if not _np.iterable(nls):
        nls = [nls]

    for i, nl in enumerate(nls):
        ax2.plot(bins, nl.torate(bins, bin_rate=bin_rate), colors[0], lw=2)

    #arange axis
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    ax1.set_xlabel('Filtered stimulus',fontsize=10)
    ax1.set_ylabel('',fontsize=10)
    ax1.yaxis.set_label_coords(-.25, .40)
    fig.subplots_adjust(bottom=.25, left=.10, right=.9, top=.95)

    #fig.set_size_inches(2, 1.5)
    fig.savefig('Figures/gating_vs_FEM_Word_cond_info.pdf', transparent=True)

    #ax1.set_xticks((-4, -2, 0, 2, 4))
    #ax1.set_xticklabels((-4, "", 0, "", 4), fontsize=10)
    #ax.set_xlim(-1,1)
    ymax = ax1.get_ylim()[1]
    ax1.set_yticks((0, ymax))
    ax1.set_yticklabels((0,ymax), fontsize=10)

    ax1.text(0.2, 4, r'$t=-100ms$',color='b')
    ax1.text(0.2, 3, r'$t= 100ms$', color='r')

    ax2.set_ylim(0, ax2.get_ylim()[1])
    #ax.legend(bbox_to_anchor=(1.4,.75), fontsize=10, handlelength=.5, frameon=False)
    fig.savefig('Figures/LP_and_sigmoids.pdf', transparent=True)
    
    return fig


def plot_compare_nls(letter_length):
    '''
    Figure to compare I(g(t0); g(t0)+n | all other letters in the word)
    
    output:
    -------
        Generates and saves plot 'Figures/compare_nls'
        
    '''
    datapath = path_template.replace('#', str(letter_length))
    
    #_pdb.set_trace()
    _plt.close('compare_nls')
    fig, ax = _plt.subplots(num='compare_nls')

    #_pdb.set_trace()
    # get all the files in current folder with MI of word and linear prediction decompossed as contributions of each letter
    basal_8L = _np.fromfile(datapath+'cond_mi_8b_0g')
    gating_8L = _np.fromfile(datapath+'cond_mi_0b_8g')

    """
    start_p = time_to_point(gating_start_t, 0)
    end_p = time_to_point(gating_end_t, 0)

    gating = basal_8L.copy()
    gating[start_p:end_p] = gating_8L[start_p:end_p]

    # get the tax if it doesn't exist
    #ax.plot(tax[:start_p], gating[:start_p], 'r', lw=2)
    #ax.plot(tax[start_p-1:end_p+1], gating[start_p-1:end_p+1], 'b', lw=2)
    #ax.plot(tax[end_p:], gating[end_p:], 'r', lw=2)
    """
    tax = _get_ploting_TAX(letter_length/1000)
    ax.plot(tax, basal_8L+basal_8L.max()/100, ':r', lw=2, label = r'$basal$')
    ax.plot(tax, gating_8L, 'b', lw=2, label=r'$gating$')

    #ax.plot(tax, gating, 'b', lw=2, label=r'$gating$')
    #ax.plot(tax, cond_mi_4L, '-.r', lw=2, label='4L')
    #ax.plot(tax, cond_mi_8L, '--r', lw=2, label='8L')


    
    # add saccade dotted line
    ax.plot((0,0), ax.get_ylim(), 'k:', label='_nolegend_')

    ax.set_xlim(-.2, .8)
    ax.legend(loc='center right', fontsize=10, handlelength=1.5, handletextpad=.25, frameon=False, bbox_to_anchor=(1,.7))#, frameon=False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlabel(r'$Time\, (s)$', fontsize=12)
    ax.set_ylabel(r'$Information\, (Bits)$', fontsize=12)
    ax.yaxis.set_label_coords(-.10, .4)
    #ax.set_title('g( g(t)+n; g(t) | all other letters)\ncomparing different word lengths')
   
    xticks = _np.arange(-.2, .8, .4)
    yticks = range(0, int(ax.get_ylim()[1]), 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, size=10)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, size=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3)

    # add margin for axis labels
    fig.subplots_adjust(bottom=.2, left=.2, right=1, top=1)
    fig.set_size_inches(2.5, 2)
    fig.savefig(datapath + 'compare_nls.pdf', transparent=True)

    return fig

def plot_letter_length():
    '''
    Figure to compare I(g(t0);g(t0)+n | all ohter letters) for words of a fixed number of letters, where the letter_length changes

    output:
    -------
        generates and saves plot 'Figures/letter_length'
    '''

    _plt.close('letter_length')
    fig, ax = _plt.subplots(num='letter_length')

    trace_5ms = _np.fromfile(datapath+'cond_mi_2b_0g_5ms')
    trace_25ms = _np.fromfile(datapath+'cond_mi_2b_0g_25ms')
    trace_125ms = _np.fromfile(datapath+'cond_mi_2b_0g_125ms')

    tax = _get_ploting_TAX(letter_length/1000)
    ax.plot(tax, trace_5ms, 'r', linewidth=2, label='5ms')
    ax.plot(tax, trace_25ms, '-.r', linewidth=2, label='25ms')
    ax.plot(tax, trace_125ms, ':r', linewidth=2, label='125ms')

    _plt.plot((0,0), (0, ax.get_ylim()[1]), 'k:', label='_nolabel_')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig


def plot_gating_effect(letter_length):
    from os import listdir

    # get the list of files in datapath+'' with gating_effect_2L
    gating_file = [name for name in listdir(datapath+'') if name.startswith('gating_effect_2L')][0]
    nogating_file = [name for name in listdir(datapath+'') if name.startswith('nogating_effect_2L_{0}'.format(letter_length))][0]

    gating = _np.fromfile(datapath+''+gating_file)
    nogating = _np.fromfile(datapath+''+nogating_file)

    tax = _get_ploting_TAX(letter_length/1000)
    _plt.close('gating_effect')
    fig, ax = _plt.subplots(num='gating_effect')

    ax.plot(tax, gating, 'b', lw=2)
    ax.plot(tax, nogating, 'r', lw=2)

    ax.set_xlim(-.2, .8)

    fig.set_size_inches(2, 2)
    fig.savefig('Figures/gating_effect.pdf', transparent=True)

    return fig

def plot_densities(letter_length):
    '''
    Plot histograms after binning the data (g, basal and gating)
    '''
    datapath = path_template.replace('#', str(letter_length))
    sacc_path = 'Sacc={0}, RWS=.0{1}/'.format(saccade_size, int(rw_step*100))
    
    #_pdb.set_trace()

    # Load g and letters before binning
    g_pre       = _np.fromfile(sacc_path+'g_preSac_nobinning')
    g_post      = _np.fromfile(sacc_path+'g_postSac_nobinning')
    basal_pre   = _np.fromfile(sacc_path+'basal_letters_preSac_nobinning')
    basal_post  = _np.fromfile(sacc_path+'basal_letters_postSac_nobinning')
    gating_pre   = _np.fromfile(sacc_path+'gating_letters_preSac_nobinning')
    gating_post  = _np.fromfile(sacc_path+'gating_letters_postSac_nobinning')

    _plt.close('densities')
    fig, ax = _plt.subplots(nrows=3, num='densities')

    _, bins, _= ax[0].hist(g_post, bins=200, normed=True, histtype='stepfilled', alpha=.5, label=r'$lp, t=100ms$')
    ax[0].hist(g_pre, bins=bins, normed=True, histtype='stepfilled', alpha=.5, label=r'$lp, t=-100ms$')
    _, bins, _ = ax[1].hist(gating_pre, bins=200, normed=True, histtype='step', lw=2, alpha=1, label=r'$gating, t=-100ms$')
    ax[1].hist(basal_pre, bins=bins, normed=True, histtype='step', lw=2, alpha=1, label=r'$basal, t=-100ms$')
    _, bins, _ = ax[2].hist(gating_post, bins=200, normed=True, histtype='step', lw=2, alpha=1, label=r'$gating, t=100ms$')
    ax[2].hist(basal_post, bins=bins, normed=True, histtype='step', alpha=1, lw=2, label=r'$basal, t=-100ms$')

    #ax[0].text(500, .008, 'LP')
    #ax[1].text(300, .1, r'$pre$')
    #ax[2].text(300, .05, r'$post$')

    ax[0].legend(loc='upper right', fontsize=10, handlelength=1.5)
    ax[1].legend(loc='upper right', fontsize=10, handlelength=1.5)
    ax[2].legend(loc='upper right', fontsize=10, handlelength=1.5)

    ax[1].set_xlim(0,300)
    ax[2].set_xlim(0,300)
    ax[2].set_ylim(0, .03)

    fig.savefig(datapath + 'densities.pdf', transparent=True)

    return fig

def plot_binned_density(letter_length):
    '''
    Plot histograms after binning the data (g, basal and gating)
    '''
    datapath = path_template.replace('#', str(letter_length))
    
    #_pdb.set_trace()

    # load binned data, I only have two slices of time for each (at -.1 and .1 secs relative to saccade)
    binned_g_pre        = _np.fromfile(datapath + 'binned_g_preSac', dtype=int)
    binned_basal_pre    = _np.fromfile(datapath + 'binned_basal_preSac', dtype=int)
    binned_gating_pre    = _np.fromfile(datapath + 'binned_gating_preSac', dtype=int)
    binned_g_post       = _np.fromfile(datapath + 'binned_g_postSac', dtype=int)
    binned_basal_post   = _np.fromfile(datapath + 'binned_basal_postSac', dtype=int)
    binned_gating_post   = _np.fromfile(datapath + 'binned_gating_postSac', dtype=int)

    _plt.close('binned_density')
    fig, ax = _plt.subplots(nrows=3, num='binned_density')

    hatch = ('xxx', None, 'x', '-','o','/','\\',None, '*','\/')
    _, bins, _ = ax[0].hist([binned_g_post, binned_g_pre], bins=binsN, color='gy', normed=True, histtype='bar', alpha=.6, label=[r'$t=100ms$', r'$t=-100ms$'])
    #ax[0].hist(binned_g_pre, bins=bins, normed=True, histtype='stepfilled', alpha=.5, label=r'$lp, t=-100ms$')

    #_pdb.set_trace()
    _, bins, patches = ax[1].hist([binned_basal_pre, binned_gating_pre], color = 'yy', bins=binsN, normed=True, histtype='bar', alpha=.8, label=[r'$basal$', r'$gating$'])
    for i, patch in enumerate(patches):
        for bar in patch:
            bar.set_hatch(hatch[i])


    _, bins, patches = ax[2].hist([binned_basal_post, binned_gating_post], bins=binsN, color='gg', normed=True, histtype='bar', alpha=.8, label=[r'$basal$', r'$gating$'])
    for i, patch in enumerate(patches):
        for bar in patch:
            bar.set_hatch(hatch[i])

    ax[0].text(2, ax[0].get_ylim()[1]*.8, r'$Linear Prediction$')
    ax[1].text(8, ax[1].get_ylim()[1]*.8, r'$t=-100ms$')
    ax[2].text(8, ax[2].get_ylim()[1]*.8, r'$t=100ms$')
    #ax[0].set_xlim(1,16)
    #ax[1].set_xlim(4,16)
    #ax[2].set_xlim(4,16)

    #ax[0].text(14, .08, 'LP')
    #ax[1].text(14, 1, r'$basal$')
    #ax[2].text(14, .5, r'$gating$')

    #xticks = range(5,17)
    #xlabels = range(0,12)
    #ax[1].set_xticklabels(xlabels, fontsize=10)
    #ax[2].set_xticklabels(xlabels, fontsize=10)

    ax[0].legend(loc='best')#upper right')#, fontsize=10, handlelength=2)
    ax[1].legend(loc='best')#'upper right')#, fontsize=10, handlelength=2)
    ax[2].legend(loc='best')#'upper right')#, fontsize=10, handlelength=2)

    fig.subplots_adjust(left=0, right=.95)
    fig.set_size_inches(10,6)
    fig.savefig(datapath + 'binned_density.pdf', transparent=True)

    return fig

def plot_letters_N(letter_length, basal_lettersN, gating_lettersN):
    '''
    plot several arrays, each containing the information the last letter carries about g given the previous letters. Each trace in the word is for a different number of letters per word.

    letter_length is an int in ms
    '''
    
    datapath = path_template.replace('#', str(letter_length))
    
    #_pdb.set_trace()
    _plt.close('letters_N')
    fig, ax = _plt.subplots(num='letters_N')

    tax = _get_ploting_TAX(letter_length/1000)

    # get all the files in current folder with MI of word and linear prediction decompossed as contributions of each letter
    mode = ['r', 'r:', 'r--', 'r-.']
    for n in basal_lettersN:
        mode_index = int(_np.log2(n))
        if n == 1:
            mi = _np.fromfile(datapath+'mi_1b_0g')
        else:
            mi = _np.fromfile(datapath+'cond_mi_{0}b_0g'.format(n, letter_length))

        ax.plot(tax, mi, mode[mode_index], lw=2, label=r'${0}$'.format(n))

    mode = ['b','b:', 'b--', 'b-.']
    for n in gating_lettersN:
        mode_index = int(_np.log2(n))
        if n == 1:
            mi = _np.fromfile(datapath+'mi_0b_1g')
        else:
            mi = _np.fromfile(datapath+'cond_mi_0b_{0}g'.format(n, letter_length))

        ax.plot(tax, mi, mode[mode_index], lw=2, label=r'${0}$'.format(n))

    """
    mi_1L = _np.fromfile(datapath+'mi_1b_0g')
    cond_mi_2L = _np.fromfile(datapath+'cond_mi_2b_0g')
    cond_mi_4L = _np.fromfile(datapath+'cond_mi_4b_0g')
    cond_mi_8L = _np.fromfile(datapath+'cond_mi_8b_0g')

    ax.plot(tax, mi_1L, 'r', lw=2, label=r'$1L$')
    ax.plot(tax, cond_mi_2L, ':r', lw=2, label=r'$2L$')
    #ax.plot(tax, cond_mi_4L, ':r', lw=2, label='4L')
    #ax.plot(tax, cond_mi_8L, ':r', lw=2, label='8L')
    """

    ax.plot([0,0], ax.get_ylim(), ':k', label='_nolegend_')

    ax.legend(fontsize=10, handlelength=1.5, frameon=False)

    xticks = _np.arange(-.2, .8, .4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)
    yticks = range(0, int(ax.get_ylim()[1]+1), 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=10)
    ax.set_xlim(-.2, .8)

    ax.tick_params(right='off', top='off', length=3)
    ax.set_xlabel(r'$Time\,(s)$')
    ax.set_ylabel(r'$Informaton\,(Bits)$')
    

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.subplots_adjust(left=.2, bottom=.3, right=1, top=1)
    fig.set_size_inches(2.5, 2)
    fig.savefig(datapath + 'letters_N.pdf', transparent=True)

    return fig

def plot_letters_N2(letter_length, basal_flag=1):
    '''
    plot several arrays, each containing the information the last letter carries about g given the previous letters. Each trace in the word is for a different number of letters per word.

    letter_length is an int in ms

    basal_flag      = 0,    plots basal cond info
                    = 1,    plots gating cond info
                    
    '''
    
    datapath = path_template.replace('#', str(letter_length))
    
    #_pdb.set_trace()
    _plt.close('letters_N2')
    fig, ax = _plt.subplots(num='letters_N2')

    tax = _get_ploting_TAX(letter_length/1000)

    # get all the files in current folder with MI of word and linear prediction decompossed as contributions of each letter
    if basal_flag:
        mi_1L = _np.fromfile(datapath+'mi_1b_0g')
        cond_mi_2L = _np.fromfile(datapath+'cond_mi_2b_0g')
        cond_mi_4L = _np.fromfile(datapath+'cond_mi_4b_0g')
        cond_mi_8L = _np.fromfile(datapath+'cond_mi_8b_0g')
        gating_mi_2L = _np.fromfile(datapath+'cond_mi_0b_2g')
    else:
        mi_1L = _np.fromfile(datapath+'mi_0b_1g')
        cond_mi_2L = _np.fromfile(datapath+'cond_mi_0b_2g')
        cond_mi_4L = _np.fromfile(datapath+'cond_mi_0b_2g')
        cond_mi_8L = _np.fromfile(datapath+'cond_mi_0b_2g')

    # get the tax if it doesn't exist
    tax = _get_ploting_TAX(letter_length/1000)
    ax.plot(tax, mi_1L, lw=2, label=r'$1L$')
    ax.plot(tax, cond_mi_2L, lw=2, label=r'$2L$')
    ax.plot(tax, cond_mi_4L, lw=2, label=r'$4L$')
    ax.plot(tax, cond_mi_8L, lw=2, label=r'$8L$')
    ax.plot(tax, gating_mi_2L, lw=2)

    ax.plot([0,0], ax.get_ylim(), ':k', label='_nolegend_')

    ax.legend(fontsize=10, handlelength=1.5, frameon=False,bbox_to_anchor=(1.05,1.2))

    xticks = _np.arange(-.2, .8, .4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)
    yticks = range(0, int(ax.get_ylim()[1]+1), 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=10)
    ax.set_xlim(-.2, .8)

    ax.tick_params(right='off', top='off', length=3)
    ax.set_xlabel(r'$Time\,(s)$')
    ax.set_ylabel(r'$Informaton\,(Bits)$')
    

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.subplots_adjust(left=.2, bottom=.2, right=1, top=.9)
    fig.set_size_inches(2.5, 2)
    fig.savefig(datapath + 'letters_N2.pdf', transparent=True)

    return fig


def plot_pink_stim():
    '''
    Plot a vew seconds of pink stimulus
    '''
    length = 10  # in seconds
    length /= sim_delta_t   # in samples
    length /= _np.round(.03/sim_delta_t)
    stim = _pn.pink(int(length)).reshape(-1, 1)*_np.ones((1, _np.round(.03/sim_delta_t)))
    stim = stim.reshape(-1,1)
    _plt.close('pink_stim')
    fig, ax = _plt.subplots(num='pink_stim')
    
    ax.plot(stim, 'k')

    fig.savefig('Figures/pink_stim.pdf', transparent=True)

    return fig


def plot_compare_letter_length(letter_length_list):
    ''' 
    for the given letters, compare some type of curve, either the conditional information with 2L or the total information
    '''

    #_pdb.set_trace()
    try:
        _plt.close('compare_letter_length')
        fig, ax = _plt.subplots(num = 'compare_letter_length')

        line_style = ['bo', 'g*', 'rs', 'mx', 'm^']
        for i, letter_length in enumerate(letter_length_list):
            datapath = path_template.replace('#', str(letter_length))

            tax = _get_ploting_TAX(letter_length/1000)
            trace = _np.fromfile(datapath + 'cond_mi_2b_0g')
            N = len(tax)
            newN = 15
            #ax.plot(tax[::N/newN], trace[::N/newN], line_style[i], markersize=4, lw=2, label = '{0}ms'.format(letter_length))
            ax.plot(tax, trace, lw=2, label = '{0}ms'.format(letter_length))

        ax.legend(loc=1, fontsize=10, handlelength=1.5, bbox_to_anchor = (1, 1))

        # add vertical dotted line at saccade
        ax.plot([0,0], ax.get_ylim(), ':k', label='_nolegend_', lw=2)

        xticks = _np.arange(-.2, .8, .4)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=10)
        ax.set_xlim(-.2, .8)

        yticks = range(0, int(ax.get_ylim()[1]+1), 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=10)
        ax.set_ylabel(r'$Information (Bits)$', fontsize=12)
        
        ax.set_xlabel(r'$Time\, (s)$')

        ax.tick_params(length=3, right='off', top='off')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.set_size_inches(2.5, 2)
        fig.subplots_adjust(left=.2, bottom=.3, right=1, top=1)


        fig.savefig('Figures/compare_letter_length.pdf', transparent=True)
    except:
        print("plot_compare_letter_length failed, most likely because it couldn't load one of the cond_mi arrays")
        _plt.close('compare_letter_length')


def plot_info_ratio(info_ratio, info_ratio_sem):
    _plt.close('info_ratio')
    fig, ax = _plt.subplots(num='info_ratio')

    igor_delta_t = 0.048
    tax = _np.arange(0, info_ratio.shape[1]*igor_delta_t, igor_delta_t)
    
    colors = 'rgc'
    for i in range(info_ratio.shape[0]):
        ax.errorbar(tax, info_ratio[i,:], yerr=info_ratio_sem[i,:], color = colors[i], lw=2)

    xticks = _np.arange(0, 0.5, 0.4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)
    yticks = range(0, 5, 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=10)
    ax.set_xlabel(r'$Time\,(s)$', fontsize=10)
    ax.set_ylabel(r'$Information\, ratio$', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3, top='off', right='off')
    fig.subplots_adjust(left=.3, bottom=.3, top=.95, right=1)
    fig.set_size_inches(2.5, 1.5)

    fig.savefig('Figures/info_ratio.pdf', transparent=True)
    return fig

def plot_compare_gating_timing(letter_length_in_ms):
    '''
    plot a comparisson between the experimental information_ratio computed in igor when there is peripheral input and no central change and the simulation with natural scenes with no peripheral input, showing that gating occurs at the same time as the central information changes.

    I'm going to make plot from -.2s, so I'll get the tax and traces from 'info_ratio' and add some points at the beginning
    '''
    # First load data from igor (Summary of TNF), will generate a plot as well
    info_ratio, info_ratio_sem, fig = load_infoRatio1()

    # get x axis from the plot made by 'load_infoRatio1' (plot is called 'info_ratio')
    exp_xdata = fig.gca().get_lines()[0].get_xdata()


    # start a new plot
    _plt.close('compare_gating_timing')
    fig, ax = _plt.subplots(num='compare_gating_timing')

    # add the 1st line from 'info_ratio to the plot I'm making
    ax.errorbar(exp_xdata, info_ratio[0,:], yerr=info_ratio_sem[0,:], color='r', lw=2)
    ax.errorbar(exp_xdata, info_ratio[1,:], yerr=info_ratio_sem[0,:], color='g', lw=2)
    ax.errorbar(exp_xdata, info_ratio[2,:], yerr=info_ratio_sem[0,:], color='c', lw=2)

    # add data prior to 0s for better displaying
    #exp_negative_xdata = exp_xdata[::-1]*-1
    #ax.plot(exp_negative_xdata, _np.ones_like(exp_negative_xdata)*info_ratio[0,0], ':r', markersize=2, lw=2)
    #ax.plot(exp_negative_xdata, _np.ones_like(exp_negative_xdata)*info_ratio[1,0], ':g', markersize=2, lw=2)
    #ax.plot(exp_negative_xdata, _np.ones_like(exp_negative_xdata)*info_ratio[2,0], ':c', markersize=2, lw=2)

    # Add the simulated line with conditional information in the 8L case under basal nonlinearity
    letter_length = letter_length_in_ms/1000
    datapath = path_template.replace('#', str(letter_length_in_ms))
    tax = _np.arange(sim_start_t, sim_end_t, letter_length)
    cond_mi_8b_0g = _np.fromfile(datapath + 'cond_mi_8b_0g'.format(letter_length_in_ms))
    ax.plot(tax[:-1], cond_mi_8b_0g[:-1], 'k', lw=2)


    # add vertical line at saccade
    ax.plot([0,0], ax.get_ylim(), ':k')

    xticks = _np.arange(-.2, 0.8, 0.4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)
    ax.set_xlim(-.2, .8)
    yticks = range(0, 5, 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=10)
    ax.set_xlabel(r'$Time\,(s)$', fontsize=12)
    ax.set_ylabel(r'$Information$', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3, top='off', right='off')
    fig.subplots_adjust(left=.2, bottom=.2, top=.95, right=1)
    fig.set_size_inches(2.5, 2)

    fig.savefig('Figures/compare_gating_timing.pdf', transparent=True)


def plot_gating_vs_letter_length(letter_length_list):
    '''
    Compare the effect of gating as a function of letter length
    
    load files of the form cond_mi_2b_0g_#ms and cond_mi_0b_2g_#ms and compute the ratio between gating and no gating and display it as a function of letter_length
    '''

    #_pdb.set_trace()
    try:
        ratios = []

        for length in letter_length_list:
            datapath = path_template.replace('#', str(length))
            basal = _np.fromfile(datapath + 'cond_mi_2b_0g'.format(length))
            gating = _np.fromfile(datapath + 'cond_mi_0b_2g'.format(length))

            #ratios.append(gating.max()/basal[gating.argmax()])
            gating_start_p = int((gating_start_t - sim_start_t)*1000/length)
            gating_end_p = int((gating_end_t - sim_start_t)*1000/length)

            ratios.append(gating[gating_start_p:gating_end_p].mean()/basal[gating_start_p:gating_end_p].mean())

        _plt.close('gating_vs_letter_length')
        fig, ax = _plt.subplots(num='gating_vs_letter_length')
        
        ax.plot(letter_length_list, ratios, 'ok')

        ax.set_xticks(letter_length_list)
        ax.set_xticklabels(letter_length_list, fontsize=10)
        #ax.set_xscale('log')
        ax.set_xlabel(r'$Letter\, length (ms)$')
        ax.set_xlim(0, letter_length_list[-1]+10)

        yticks = range(0, 3, 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=10)
        ax.set_ylabel(r'$Information\, Ratio$')
        ax.set_ylim(.5, 3)

        # Add doted line at y=1
        ax.plot(ax.get_xlim(), [1,1], ':k')

        fig.subplots_adjust(left=.2, bottom=.3, top=1, right=1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(length=3, right='off', top='off')
        fig.set_size_inches(2.5, 2)

        fig.savefig('Figures/plot_gating_vs_letter_length.pdf')

    except:
        _plt.close('gating_vs_letter_length')


def plot_integrated_information(letter_length_in_ms, letters_N_list):
    basal = []
    gating = []

    datapath = 'Data/{0}ms letter_length/'.format(letter_length_in_ms)

    # Figure out the points corresponding to t = 0s and t = inhibition_end_t
    p0 = int(-sim_start_t*1000/letter_length_in_ms)
    p1 = int((inhibition_end_t-sim_start_t)*1000/letter_length_in_ms)
    
    for N in letters_N_list:
        basal_info = _np.fromfile(datapath + 'integrated_info_{0}b_0g'.format(N))
        gating_info = _np.fromfile(datapath + 'integrated_info_0b_{0}g'.format(N))
        basal.append(basal_info[p1]-basal_info[p0])
        gating.append(gating_info[p1]-gating_info[p0])

    # make the plot
    _plt.close('integrated_information')
    fig, ax = _plt.subplots(num='integrated_information')

    ax.plot(letters_N_list, basal, 'ro', label=r'$basal$')
    ax.plot(letters_N_list, gating, 'bo', label=r'$gating$')

    xticks = [1,2,4,8]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)
    ax.set_xlabel(r'$Letters N$')
    ax.set_xlim(0, 9)

    yticks = range(0, 10, 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=10)
    ax.set_ylabel(r'$Information$')
    ax.set_ylim(3.5, 6.5)

    ax.text(5, 5.8, r'$basal$', color='r')
    ax.text(5, 5.2, r'$gating$', color='b')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3, right='off', top='off')
    fig.subplots_adjust(left=0.3, bottom=.3, top=1, right=1)
    fig.set_size_inches(2.5, 1.5)

    fig.savefig(datapath + 'integrated_information.pdf', transparent=True)

def plot_information_delivery_time(letter_length_in_ms, letters_N_list, threshold):
    _pdb.set_trace()
    basal = []
    gating = []

    for N in letters_N_list:
        times = _get_information_delivery_time(letter_length_in_ms, N, threshold)
        basal.append(times[0])
        gating.append(times[1])

    # make the plot
    _plt.close('information_delivery_time')
    fig, ax = _plt.subplots(num='information_delivery_time')

    ax.plot(letters_N_list, basal, 'ro', label=r'$basal$')
    ax.plot(letters_N_list, gating, 'bo', label=r'$gating$')

    xticks = [1,2,4,8]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)
    ax.set_xlabel(r'$Letters N$')
    ax.set_xlim(0, 9)

    yticks = _np.arange(0.05, 0.4, 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=10)
    ax.set_ylabel(r'$Delivery\, Time$')
    ax.set_ylim(0, .4)

    #ax.legend(fontsize=10, bbox_to_anchor=(0.75, 1.2))
    ax.text(5,.35, r'$basal$', color='r')
    ax.text(5,.27, r'$gating$', color='b')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3, right='off', top='off')
    fig.subplots_adjust(left=0.3, bottom=.3, top=.951, right=1)
    fig.set_size_inches(2.5, 1.5)

    datapath = 'Data/{0}ms letter_length/'.format(letter_length_in_ms)
    fig.savefig(datapath + 'information_delivery_time.pdf', transparent=True)

def plot_bits_per_spike(letter_length_in_ms, letters_N_list):
    '''
    bits per spike as a function of word length
    '''
    basal = []
    gating = []
    ratio = []
    for N in letters_N_list:
        info_per_spike = get_info_per_spike(letter_length_in_ms, N)
        basal.append(info_per_spike[0])
        gating.append(info_per_spike[1])
        ratio.append(info_per_spike[1]/info_per_spike[0])

    _plt.close('bits_per_spike')
    fig, ax = _plt.subplots(num='bits_per_spike')

    ax.plot(letters_N_list, ratio, 'ko', label='basal')
    #ax.plot(letters_N_list, basal, 'ro', label='basal')
    #ax.plot(letters_N_list, gating, 'bo', label='gating')


    xticks = [1,2,4,8]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=10)
    ax.set_xlabel(r'$Letters N$')
    ax.set_xlim(0, 9)

    yticks = _np.arange(0.5, 1.5, 0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=10)
    ax.set_ylabel(r'$\frac{gating\, efficiency}{basal\, efficiency}$')
    ax.set_ylim(0.4, 1.4)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3, right='off', top='off')
    fig.subplots_adjust(left=0.3, bottom=.3, top=1, right=1)
    fig.set_size_inches(2.5, 1.5)


    datapath = 'Data/{0}ms letter_length/'.format(letter_length_in_ms)
    fig.savefig(datapath + 'bits_per_spike.pdf', transparent=True)


def plot_adaptive_index_slopes(contrasts, slopes, tw_list):
    '''
    slopes[i,j] is the gain of a nl for a given contrast i at tw j.
    '''

    _plt.close('adaptive_index_slopes')
    fig, ax = _plt.subplots(num = 'adaptive_index_slopes')

    #_pdb.set_trace()
    for i, tw in enumerate(tw_list):
        max_slope = slopes[:, tw].max()

        xticks = _np.divide(1, contrasts)
        ax.plot(xticks, slopes[:, tw]/max_slope)

        ax.set_xticks(xticks)
        xlabels = [r'$\frac{1}{'+str(c)+'}$' for c in contrasts]
        print(xlabels)
        ax.set_xticklabels(xlabels, fontsize=14)
# plots go here

def information(cov, X):
    '''
    cov is the covariance matrix of the simulation. Each point along either of the two axis represents a point in time (from sim_start_t to sim_end_t in steps of length sim_delta_t)
    Here, produce a 1D ndarray that at each point 'p' (corresponding to time startT + p*deltaT) , computes the mutual information between the linear prediction at points X (relative to each p) and p.
    Coputes MI(X+p, p) and here the +P is not list concatenation but item by item summation
    For example: information(cov, [-1, -2, -3]), computes for every point p: infomration(cov, [-1+p, -2+p, -3+p], p)
    
    inputs:
    -------
        cov:    2D ndarray, covariance matrix of the linear prediction

        X:      iterable of ints, points relative to each point in the time axis to compute MI with

    outputs:
    --------
        information     1D ndarray, each point has the information between that point and X
                        size of output is same as cov.shape[0]
                        
    '''

    # first lets convert X to ndarray it it is not
    if not isinstance(X, _np.ndarray):
        X = _np.array(X, dtype='int')
    
    return _np.array([_info.gaussianInformation(cov, X+p, [p]) if p+X.min()>=0 and p+X.max()<cov.shape[0] else _np.nan for p in range(cov.shape[0])])
    #return [(X+p, p) if p+X.min()>=0 and p+X.max()<cov.shape[0] else _np.nan for p in range(cov.shape[0])]

def _getCondInfoP0(cov, covN, p0, condListLP, condListLPN):
    '''
    Compute the conditional information between the LP and the LP + noise at time corresponding to point p0, conditional on all time points from condListLP and condListLPN

    Computes:
        I(g(t); g(t)+n | g(ta_0), ..., g(ta_n), g(tb_0)+noise, ..., g(tb_n)+noise)

        where ta_0, ... ,ta_n are times corresponding to points in condListLP and tb_0, ..., tb_n are times corresponding to points in condListLPN

    inputs:
    -------
        cov:    2D ndarray, covariance matrix of the linear prediction, comres from the simulation

        covN:   2D ndarray, covariance matrix of the noise, comes from the variance in the simulation and Yusuf's intracellular data

        p0:     int, point corresonding to time in g(t) and g(t)+n in the calculation
                p0 = (t - sim_start_t)/sim_delta_t

        condListLP:     list of ints. Point is relative to p0. -1 represents sim_delta_t prior to p0, etc.
                        allows to condition the information calculation at time t on g(t0), g(t1), etc whrere t0, t1, correspond to points in condListLP
        
        condListLPN:    idem condListLP but conditions on g(t0)+noise for all t0 in condListLPN

    output:
    -------
        info:   1D ndarray, the conditional mutual information

    Implemenation notes:
        I(X; Y | Z) = H(X | Z) - H(X | Y, Z)
                    = H(X, Z) - H(Z) - ( H(X, Y, Z) - H(Y, Z) )
                    = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)


        In computing all these entropies, I will generate the covariance matrix between LP and LP + Noise for the time points requested. The covariance between different time points of LP is just a submatrix of LP. The covariance matrix between different time points of LP + noise is a submatrix of covarianceLP + the corresponding diagonal terms from covN. The covariance matrix between g(t0) and g(t1)+N is a submatrix of cov (at t0, and t1) with noise from covN added to t1
    '''
    if not isinstance(condListLP, list):
        condListLP = list(condListLP)
    if not isinstance(condListLPN, list):
        condListLPN = list(condListLPN)

    # According to the implementaion note, I will need to compute 4 different entropies. Extract the subCov corresponding to each one of them
    XZ = _extractSubCov(cov, covN, [p0] + condListLP, condListLPN)
    YZ = _extractSubCov(cov, covN, condListLP, [p0] + condListLPN)
    XYZ = _extractSubCov(cov, covN, [p0] + condListLP, [p0] + condListLPN)
    Z = _extractSubCov(cov, covN, condListLP, condListLPN)
    
    if len(Z)==0:
        # Special case, only used if not conditioning on anything, it actually computes the I(X,Y) sinze Z is empty
        return _info.gaussianEntropy(XZ) + _info.gaussianEntropy(YZ) - _info.gaussianEntropy(XYZ)
    else:
        return _info.gaussianEntropy(XZ) + _info.gaussianEntropy(YZ) - _info.gaussianEntropy(XYZ) - _info.gaussianEntropy(Z)

def _extractSubCov(covLP, covN, noiselessPoints, noisyPoints):
    '''
    Given the covariance for the LP and the covariance of the noise, extract a new covariance matrix that corresponds to noiseless and noisyPoints
    
    Assume that there are 'A' noiseless points and 'B' noisy points.

    The output is a covariance matrix of dimension (A+B) x (A+B), the interaction among points in this submatrix might be due to correlations in the linear prediction or in the noise.

    inputs:
    -------
        covLP:  2D ndarray, covariance matrix of the linear prediction, comes from the simulation

        covN:   2D ndarray, covariance matrix of the noise, comes from the variance in the simulation and Yusuf's intracellular data

        noiselessPoints:     list of ints. Each point corresponds to g(t) through p=(t-sim_start_t)/sim_delta_t
        
        noisyPoints:         list of ints. Each point corresponds to g(t)+noise through p=(t-sim_start_t)/sim_delta_t

    output:
    -------
        subCov:   2D ndarray, the covariance matrix of the points choosen

        I will generate the covariance matrix between LP and LP + Noise for the time points requested. The covariance between different time points of LP is just a submatrix of LP. The covariance matrix between different time points of LP + noise is a submatrix of covarianceLP + the corresponding diagonal terms from covN. The covariance matrix between g(t0) and g(t1)+N is a submatrix of cov (at t0, and t1) with noise from covN added to t1 and no off diagonal element because I'm assuming that noise is uncorrelated in time.
    '''

    allPoints = noiselessPoints + noisyPoints

    if allPoints==[]:
        return _np.array([])

    # extract the subarray corresponding to allPoints from covLP. I'm taking points with 'take' from a flatten version of cov. At this point in the code, there is no reference to the noise
    from itertools import product
    subCovG = _np.array([_np.take(covLP.flatten(), coord[0]+covLP.shape[0]*coord[1]) for coord in product(allPoints, allPoints)]).reshape(-1, len(allPoints))
    #print(subCov)
    
    # In delaing with noisy point I replace all noiseless points by None such that testing becomes simple. Now after product if any of the points comes from noiselessPoints, the test "None in coord" will return True
    allPoints = [None]*len(noiselessPoints) + noisyPoints
    subCovN = _np.array([0 if None in coord else _np.take(covN.flatten(), coord[0]+covN.shape[0]*coord[1]) for coord in product(allPoints, allPoints)]).reshape(-1, len(allPoints))
    
    return subCovG + subCovN
    '''
    # add noise in the diagonal terms corresponding to noisyPoints. Since I'm assuming that different time points have noise that is uncorrelated, then there are no contributions to off diagonal terms, unless there is a point more than once in noisyPoints. In that case, the contribution to the off diagonal term is covN[p,p]
    #_pdb.set_trace()
    N0 = len(noiselessPoints)   # all N0 first dimensions of subCov correspond to noiseless points
    N1 = len(noisyPoints)       # then, there are N1 dimesnions with noise.

    for i, coords in enumerate(product(noisyPoints, noisyPoints)):
        if coords[0]==coords[1]:
            i0 = _np.mod(i, N1)+N0
            i1 = _np.floor(i/N1)+N0
            subCov[i0, i1] += covN[coords]
    '''


def get_info_per_spike(letter_length_in_ms, letters_N):
    '''
    Compute total information due to teh saccade in between gating_start_t and inhibition_end_t and divide by the firing rate in the same time window
    '''
    
    #_pdb.set_trace()
    datapath = 'Data/{0}ms letter_length/'.format(letter_length_in_ms)

    # access the integrated information arrays
    try:
        int_info_basal = _np.fromfile(datapath + 'integrated_info_{0}b_0g'.format(letters_N))
        int_info_gating = _np.fromfile(datapath + 'integrated_info_0b_{0}g'.format(letters_N))
    except:
        int_info_basal, int_info_gating = _integrate_information(letter_length_in_ms, letters_N)

    # access the FR
    basal_FR = _np.fromfile(datapath + 'basal_FR')
    gating_FR = _np.fromfile(datapath + 'gating_FR')

    # integrate fr and extract spike count in bewteen gating_start_t and inhibition_end_t
    basal_FR = basal_FR.cumsum()*letter_length_in_ms/1000
    gating_FR = gating_FR.cumsum()*letter_length_in_ms/1000

    # Figure out the points corresponding to t = 0s and t = inhibition_end_t
    p0 = int(-sim_start_t*1000/letter_length_in_ms)
    p1 = int((inhibition_end_t-sim_start_t)*1000/letter_length_in_ms)
    

    basal_efficiency = (int_info_basal[p1]-int_info_basal[p0])/(basal_FR[p1]-basal_FR[p0])
    gating_efficiency = (int_info_gating[p1]-int_info_gating[p0])/(gating_FR[p1]-gating_FR[p0])

    return basal_efficiency, gating_efficiency

def _integrate_information(letter_length_in_ms, letters_N):
    '''
    From the array representing information for letters_N and letter_length (both basal and gating), integrate the information due to the central change at the saccade (the difference between the computed information and that corresponding to FEM)
    '''
    #_pdb.set_trace()
    datapath = 'Data/{0}ms letter_length/'.format(letter_length_in_ms)

    # access the information arrays
    if letters_N == 1:
        basal = _np.fromfile(datapath + 'mi_1b_0g'.format(letter_length_in_ms))
        gating = _np.fromfile(datapath + 'mi_0b_1g'.format(letter_length_in_ms))
    else:
        basal = _np.fromfile(datapath + 'cond_mi_{0}b_0g'.format(letters_N, letter_length_in_ms))
        gating = _np.fromfile(datapath + 'cond_mi_0b_{0}g'.format(letters_N, letter_length_in_ms))

    # Compute average information before saccade
    pnt0 = int(-sim_start_t*1000/letter_length_in_ms)
    basal_avg   = basal[:pnt0].mean()
    gating_avg  = gating[:pnt0].mean()

    # integrate the difference
    basal -= basal_avg
    gating -= gating_avg
    basal = basal.cumsum()#*letter_length_in_ms/1000
    gating = gating.cumsum()#*letter_length_in_ms/1000

    basal.tofile(datapath + 'integrated_info_{0}b_0g'.format(letters_N))
    gating.tofile(datapath + 'integrated_info_0b_{0}g'.format(letters_N))

    _plt.close('test')
    fig, ax = _plt.subplots(num='test')
    tax = _get_ploting_TAX(letter_length_in_ms/1000)
    ax.plot(tax, basal)
    ax.plot(tax, gating)

    return basal, gating

def _get_information_delivery_time(letter_length_in_ms, letters_N, threshold):
    '''
    given the output of _integrate_information (either basal or gating), compute at what time 'threshold' information is reached.
    
    Take the information at time 0 and figure out at what time the information crosses info_at_0 + threshold*(info_at_inhibition_end_t - info_at_0)

    input:
    ------
        Threshold:      (float) should be a number between 0 and 1
    
    '''
    _pdb.set_trace()
    datapath = 'Data/{0}ms letter_length/'.format(letter_length_in_ms)

    basal = _np.fromfile(datapath + 'integrated_info_{0}b_0g'.format(letters_N))
    gating = _np.fromfile(datapath + 'integrated_info_0b_{0}g'.format(letters_N))
    
    # Figure out the points corresponding to t = 0s and t = inhibition_end_t
    point_0 = int(-sim_start_t*1000/letter_length_in_ms)
    point_1 = int((inhibition_end_t-sim_start_t)*1000/letter_length_in_ms)

    # figure out the times at which basal and gating informations crosses threshold
    basal_t = _np.where(basal > basal[point_0]*(1-threshold) + threshold*basal[point_1])[0][0]*letter_length_in_ms/1000 + sim_start_t
    gating_t = _np.where(gating > gating[point_0]*(1-threshold) + threshold*gating[point_1])[0][0]*letter_length_in_ms/1000 + sim_start_t
    
    return basal_t, gating_t

def getInfo0(covG, covN):
    '''
    compute the mutual information between a gaussian process with covariance covG and a noisy version corrupted by an additive gaussian process with covariance covN
    I( g ; g+noise) = H(g + noise) - H(g + noise | g)
                    because g  and noise are both gaussian and uncorrelated this ends up being:
                    = 0.5 * log2( 2*pi*e * (varG + varN) ) - 0.5 * log2(2*pi*e * varN)
                    = 0.5 * log2( (varG + varN)/varN)
                    = 0.5 * log2( 1 + SNR )
                    which is a well known result

    '''
    return _np.array( [0.5 * _np.log2(1 + covG[i,i]/covN[i,i]) for i in range(covG.shape[0])])
    
def getNoiseCovariance(covG, sim_noise_fit, decay_time):
    '''
    ******** Very Important *********
    * Everything is in the simulation units and not in Yusuf's units.
    *********************************
    
    From Yusuf's data, I have the noise in the Bipolar cell's membrane potential as a function of contrast (stim SD)
    I have reproduce Yusuf experiments in reproduce_Yusuf() and scaled the noise to be in simulation units.

    Here I will return the variance of the noise, given the variance in the signal
    
    Implementation Notes:
        I'm implementing noise that is correlated in time and decays exponentialy. therefore in order to compute the noise I have to do two things
        1. The diagonal elements of the noise are coming streight from Yusuf's data:
        2. The off diagonal terms are a mixture of the diagonal term noise that decays exponentialy with time.

        In computing diagonal terms of noise from yusuf data:
            covG[i,i] is the variance at point i.
            convert to SD
            use the linear fit to convert the stimulus SD into a noise SD
            convert back to a variance

    inptus:
    -------
        covG (2d ndarray):    comes from passing images that are in the range 0-255 through some filter and then computing cov.

        sim_noise_fit (poly1d object):      linear fit to sim_nosie_sd vs sim_mp_sd

        decay_time:     time points t0 and t1 have noise that is correlated according to exp(-abs(t0-t1)/decay_time)


    output:
    -------
        noise:  noise the cell would experience under such input variance
    '''
    #_pdb.set_trace()

    # if covG is a number, just compute the noise and return it
    if not _np.iterable(covG):
        return sim_noise_fit(_np.sqrt(covG))**2

    # generate the covariance matrix for the noise. same shape as covG
    covN = _np.zeros_like(covG)

    # the diagonal values are computed from Yusuf's intracellular data
    for i in range(covG.shape[0]):
        covN[i,i] = sim_noise_fit(_np.sqrt(covG[i,i]))**2

    # To speed things I'm first computing the delay in points beyond which the correlation is too small (in those cases I will ignore it). Then I'll fill covN around the diagnol (skipping diagonal terms for which I already computed the noise

    max_distance = int(-_np.log(.1)*decay_time/sim_delta_t)     #log is the natural logarithm
                                                                # .1 is a hardcoded constant signaling when exp(-t/tau) = 0.1
    #_pdb.set_trace()
    for j in range(covG.shape[0]):
        for dist in range(1, max_distance):
            if j+dist>= covG.shape[0]:
                continue

            covN[j+dist, j] = _np.sqrt(covN[j+dist,j+dist]*covN[j,j])*_np.exp(-dist*sim_delta_t/decay_time)
            covN[j, j+dist] = covN[j+dist, j]
    
    return covN

def generate_peripheral_kernel(gating_start_t, gating_end_t, points, save_flag=0, display_flag=0):
    '''
    points should be the deisred number of points for the kernel, probably the same as in the central/surround kernels to avoid problems
    '''

    #_pdb.set_trace()
    kernel = _np.zeros(points)

    # convert gating_start/end_t to points and set the values of the kernel to periphery_exc
    gating_start_p  = int(gating_start_t/sim_delta_t)
    gating_end_p    = int(gating_end_t/sim_delta_t)
    kernel[gating_start_p:gating_end_p] = periphery_exc     # periphery_exc is a parameter defined at the top of the file

    # now set all points in between gating_end_p and recovery_start_t to periphery_inh
    recovery_start_p = int(recovery_start_t/sim_delta_t)
    kernel[gating_end_p:recovery_start_p] = -periphery_inh   # periphery_inh is a parameter defined at the top of the file

    # now set all points in between recvoery_start/end_t to a line joining periphery_inh and 0
    recovery_end_p = int(recovery_end_t/sim_delta_t)
    kernel[recovery_start_p:recovery_end_p] = _np.arange(-periphery_inh, 0, (0+periphery_inh)/(recovery_end_p-recovery_start_p))

    if save_flag:
        kernel.tofile(periphery_kernel_file, sep=' ')

    if display_flag:
        _plt.close('peripheral_kernel')
        fig, ax = _plt.subplots(num='peripheral_kernel')

        tax = _np.arange(0, len(kernel)*.005, .005)
        ax.plot(tax, kernel, lw=2)

        ax.set_ylabel(r'$Peripheral input$')
        ax.set_xlabel(r'$Time\,(s)$')

    return kernel

def time_to_point(t, return_flag):
    '''
    convert time to the nearest point in the time axis
    
    inputs:
    -------
        t (float):              in seconds

        return_flag (int):      0, return point such that time at the point is less than t
                                1, round to the nearest point
    '''
    p = (t-sim_start_t)/sim_delta_t
    if return_flag:
        return int(p)
    else:
        return int(_np.floor(p))

def point_to_s(point):
    '''
    convert from point in covG or tax to time

    input:
    ------
        point (int):    the point to get the corresponding time of

    output:
    -------
        time (int):     time in s
    '''
    return _get_simulation_TAX()[point]

def _chain_rule_info(covG, covN, p0, points):
    '''
    Decompose the information according to the chain rule and return all the terms in the chain rule.
    
    The information I'm computing is:   I(g(p0) ; g(p1)+n, ..., g(pn)      where p1, p2, ..., pn are in points

    And the chain rule is I(x0, x1, ..., xn; y) = I(x0; y) + I(x1;y | x0) + I(x2;y | x0, x1) + ... + I(xn; y | x0, x1, ..., x(n-1))
    
    Implemenation notes:
        I(X; Y | Z) = H(X | Z) - H(X | Y, Z)
                    = H(X, Z) - H(Z) - ( H(X, Y, Z) - H(Y, Z) )
                    = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
    '''
    
    # allocate ndarray output
    condInfo = _np.zeros(len(points))

    # first term in the expansion is not a conditional information, but the information between the stimulus at point p0 and the 1st noisy measurement from points
    condInfo[0] = _info.gaussianInformation(_extractSubCov(covG, covN, [p0], [points[0]]), [0], [1])
    
    #_pdb.set_trace()
    for i in range(1, len(points)):
        # According to the implementaion note, I will need to compute 4 different entropies. Extract the subCov corresponding for each one of them
        # X is always the noisy contribution at points[i]
        # Y is always the noiseless contribution at p0
        # Z are the noisy points in 0:i (not counting i), therefore Z = points[:i]
        XZ = _extractSubCov(covG, covN, [], points[:i+1])
        YZ = _extractSubCov(covG, covN, [p0], points[:i])
        XYZ = _extractSubCov(covG, covN, [p0], points[:i+1])
        Z = _extractSubCov(covG, covN, [], points[:i])
        
        condInfo[i] = _info.gaussianEntropy(XZ) + _info.gaussianEntropy(YZ) - _info.gaussianEntropy(XYZ) - _info.gaussianEntropy(Z)
        
    return condInfo

def newInformation(covG, covN, letter_length):
    '''
    compute the mutual information between noiseless sample at point p0 and the noisy sample at point p0 conditioning on all previous noisy samples


    implementation notes:
        for each point p0, compute I(g(tn); t(tn)+n | g(t0)+n, g(t1)+n, ..., g(tn)+n)

        I(X; Y | Z) = H(X | Z) - H(X | Y, Z)
                    = H(X, Z) - H(Z) - ( H(X, Y, Z) - H(Y, Z) )
                    = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
    
    where X: noiseless measurement at point pn
    Y:       noisy measurement at point pn
    Z:       noisy measurement at all points prior to pn

    in this case H(Y, Zn) = H(Z(n+1))

    and H(X, Y, Zn) = H(X, Z(n+1))

    Therefore it is faster to first compute all timepoints of both type of entropies and then combine them
    '''
    
    #_pdb.set_trace()

    Zentropy = _np.zeros(covG.shape[0])
    XZentropy = _np.zeros(covG.shape[0])
    
    for p in range(covG.shape[0]):
        Zentropy[p] = _info.gaussianEntropy(_extractSubCov(covG, covN, [], list(range(p+1))))
        XZentropy = _info.gaussianEntropy(_extractSubCov(covG, covN, [p], list(range(p+1))))


def _get_words(letter_times, gating_start_t, gating_end_t, g, covG, nogating_nl, gating_nl, gating_flag, binsN=None):
    '''
    form words by pasting together letters. 
    Gating starts and ends at times described by gating_start/end_t, outside these times, both gating and no gating use the same letters and words
    Words will be identical if theyn don't overlap with the gating window.
    
    inputs:
    -------
        letter_times (iterable of floats):
            all the times that make a word, for example: [-1.02, -1.00] for a two letter word with last letter 1s before the saccade
                                                         [0.05, .07, .09] for a three letter word

        gating_start_t (float):

        gating_end_t (float):

        g (2D ndarray):             the linear predictions for all cells

        gating_nl:                  nonlinear_block object

        gating_flag (bool):         if True and letter_times[-1] in between gating_start_t and gating_end_t uses gating_sig
                                    if False uses nogating_sig

        binsN (int):                responses are discretized from 0 to binsN-1

    outputs:
    --------
        gating_words (2D ndarray):  for each cell, returns all letters at the requested 'letter_times' times using gating sigmoid where it corresonds.
    '''

    #_pdb.set_trace()
    # allocate memory for gating_word
    words = _np.zeros((g.shape[0], len(letter_times)))
    
    # Compute maximum FR when nonlinearities are just a rectification (assuming slope of 1)
        # maxFR comes by computing max g and then subtracting the threshold
    points = [time_to_point(t) for t in letter_times]
    max_g = g[:, points].max()
    max_fr = max_g - min(nogating_sig[0], gating_sig[0])
    
    nogating_binsN = binsN * max_fr/(max_g-nogating_sig[0])
    gating_binsN = binsN * max_fr/(max_g-gating_sig[0])

    #max_fr = max(gating_sig[0]+gating_sig[1], nogating_sig[0]+nogating_sig[1])
    #nogating_binsN =  round(binsN*(nogating_sig[0]+nogating_sig[1])/max_fr)
    #gating_binsN = round(binsN*(gating_sig[0]+gating_sig[1])/max_fr)
    print('gating_binsN = {0}, nogating_binsN = {1}'.format(gating_binsN, nogating_binsN))
    for i, time in enumerate(letter_times):
        point = time_to_point(time, 0)

        if gating_flag and gating_start_t <= time and time < gating_end_t:
            words[:,i] = gating_nl.torate(g[:,point])
            binsN = gating_binsN
        else:
            words[:,i] = nogating_nl.torate(g[:,point])
            binsN = nogating_binsN

    words[:,i] = _info.binned(words, binsN, 0)

    return words

def get_total_info_since_t0(binned_g, letters, t0):
    '''
    For every time point t >= t0, compute I(g(t); letters(t) | all g's and letters between to and t)
    
    This is the total information that a system accumulates over time
    '''

    #_pdb.set_trace()
    p0 = time_to_point(t0,0)
    #delta_p = time_to_point(letter_length,0)
    sub_g = ()
    sub_L = ()

    total_info = _np.zeros(len(binned_g))

    delta_p=1
    for p in range(p0, len(binned_g), delta_p):
        sub_g = sub_g + (binned_g[p],)
        sub_L = sub_L + (letters[p],)
        tup_g = tuple(zip(*sub_g))
        tup_L = tuple(zip(*sub_L))

        total_info[p] = _info.mi(tup_g, tup_L)

    return total_info


def get_word_cond_info(g,nls):
    ''' 
    wrapper to call _get_word_cond_info

    input:
    ------
        g

        nls:        iterable of nonlinear_block objects
    
    output:
    -------
        nogating_info

        nogating_tax

        nogating_rate

        gating_info

        gating_tax

        gating_rate

        generates and saves plot Figures/gating_vs_FEM_Word_cond_info
    '''
    #_pdb.set_trace()
    letter_length = .02
    gating_start_t = .05
    gating_end_t = .15
    binsN = 32

    nogating_sig = sigmoids[0]
    gating_sig = sigmoids[1]

    tax = _np.arange(-.25, .55, letter_length)
    if nogating_sig[3]==0:
        nogating_info, nogating_rate = _np.zeros_like(tax), _np.zeros_like(tax)
    else:
        nogating_info, nogating_rate = _get_word_cond_info(letter_length, tax, gating_start_t, gating_end_t, g, covG, nogating_sig, gating_sig, binsN, 0)

    if gating_sig[3]==0:
        gating_info, gating_rate = _np.zeros_like(tax), _np.zeros_like(tax)
    else:
        gating_info, gating_rate = _get_word_cond_info(letter_length, tax, gating_start_t, gating_end_t, g, covG, nogating_sig, gating_sig, binsN, 1)
   
    return tax, nogating_info, nogating_rate, gating_info, gating_rate

def _get_word_cond_info(letter_length, tax, gating_start_t, gating_end_t, g, covG, nogating_sig, gating_sig, binsN, gating_flag):
    '''
    I'm only implementing a 2 letter word.  At each point in time, compute I( last letter of response ; g | previous letter of response)

    '''
    info = _np.zeros(len(tax))
    rate = _np.zeros(len(tax))

    for i, t in enumerate(tax):
        prev_t = t - letter_length

        binned_g = _info.binned(g[:, time_to_point(t,0)], binsN, 1)
        words = _get_words([prev_t, t], gating_start_t, gating_end_t, g, covG, nogating_sig, gating_sig, gating_flag, binsN=binsN)
        
        rate[i] = words.mean()
        info[i] = _info.cond_mi(words[:,1], binned_g, words[:,0])
    return info, rate


def info_between_fr_and_g(firing_rate, g, t1, binsN=8):
    '''
    Given the firing rate at a given time and the linear prediction (g), compute the information between the firing_rate and the linear prediction at time t1


    Compute cor(g(t>t0); firing_rate(t0))

    inputs:
    -------
        firing_rate (ndarray):  Firing rate at time t0, probably the output of convert_to_firing_rate

        g (2D ndarray):         linear prediction, simulation's output
        
        t1 (float):             correlation between firing rate and g will be computed for all times in between t0 and t1 (inclusive)
    '''

    # convert t1 to point in g
    p1 = time_to_point(t1, 0)
    
    #_pdb.set_trace()
    # bin firing rate and g using binsN
    binnedFR = _info.binned(firing_rate, binsN, 0)
    binned_g = _info.binned(g[:,p1], binsN, 1)

    return _info.mi(binnedFR, binned_g)
    #cov = _np.cov(firing_rate, g[:, p1])
    #return _info.gaussianInformation(cov, [0], [1])

def fit_exp_to_simulation(g, df, nogating_t, gating_t, cell=None):
    '''
    Load the sigmoidal nonlinearities form the experiments (during gating and FEM) and change their scaling to match the simulation
    
    inputs:
    -------
        g:                  the linear predictions from the simulation

        df (pandas df):     a data frame with all nonlinearities for all cells. Most likely the output of "load_sigmoids(length=100)"

        nogating_t (float):      time at which to fit experimental nogating_sig

        gating_t (float):   time at which to fit expeirmental gating_sig
    
        cell (int):         which cell's NL to load, has to be in the range of the df
                            if not given a random one is picked

    outputs:
    --------
        nogating_sig:            parameters for the sigmoidal nonlinearity

        gating_sig:         idem nogating_sig


    Experimental sigmoids were calculated in igor and are in units of contrast (1 in the x axis means 1 standard deviation)
    sigmoids are given with 7 numbers and they represent:
    sig[0] + sig[1]/(1+exp(-(x-sig[2])/sig[3]))
    sig[4]:     leftx of nonlinearity
    sig[5]:     rightx of nonlinearity
    sig[6]:     experimental contrast used
    '''

    """
    # Load sigmoids from 3%
    exp_nogating_sig = _np.fromfile('d100928_R1_c19_0_NL_0Y.txt', sep='\r')
    exp_gating_sig = _np.fromfile('d100928_R1_c19_0_NL_1Y.txt', sep='\r')
    """

    #_pdb.set_trace()

    if cell is None:
        cell = _np.random.randint(len(df.index))
        print('fitting sigmoids for cell in df.iloc = {0}'.format(cell))

    # even though I'm only going to use in this script threshold and sd of sigmoid (points 2 and 3) I get out of the sigmoid all 4 points to avoid confusions later on when calling w[2] and w{3]
    exp_nogating_sig = df.iloc[cell][['TW0_w[0]', 'TW0_w[1]', 'TW0_w[2]', 'TW0_w[3]']].values

    if df['gatingTW'][cell] == 1:
        exp_gating_sig = df.iloc[cell][['TW1_w[0]', 'TW1_w[1]', 'TW1_w[2]', 'TW1_w[3]']].values
    elif df['gatingTW'][cell] == 2:
        exp_gating_sig = df.iloc[cell][['TW2_w[0]', 'TW2_w[1]', 'TW2_w[2]', 'TW2_w[3]']].values
    elif df['gatingTW'][cell] == 3:
        exp_gating_sig = df.iloc[cell][['TW3_w[0]', 'TW3_w[1]', 'TW3_w[2]', 'TW3_w[3]']].values

    # find out the SD of the linear prediction at the given times
    nogating_SD = g[:, time_to_point(nogating_t,0)].std()
    gating_SD = g[:, time_to_point(gating_t, 0)].std()

    nogating_sig = exp_nogating_sig
    gating_sig = exp_gating_sig

    # if sigmoid's std is zero, just let it be, don't divide by 0 creating nans
    if exp_nogating_sig[3] != 0:
        nogating_sig[2:4] *= nogating_SD/exp_nogating_sig[3]
    
    if exp_gating_sig[3] != 0:
        gating_sig[2:4] *= gating_SD/exp_gating_sig[3]

    """
    print('printing FEM results')
    print(exp_nogating_sig)
    print(nogating_SD)
    print(nogating_sig)
    print('\r')
    print('printing Gating results')
    print(exp_gating_sig)
    print(gating_SD)
    print(gating_sig)
    """
    return nogating_sig, gating_sig


def test_several_sigmoids(g, covG, df, n=None):
    '''
    compute "get_word_cond_info" for "n" randomly choosen cells from all cells in df (a pandas data frame). If n is none it just computes it across all cells in df
    '''
    from time import time

    #_pdb.set_trace()

    if n is None:
        n_list = range(len(df.index))
    else:
        n_list = _np.random.randint(0, len(df.index), n)

    cond_info = []
    for i, n in enumerate(n_list):
        t0 = time()
        print("processing cell {0}".format(n))
        cond_info.append(get_word_cond_info(g, covG, fit_exp_to_simulation(g, df, -.1, .1, cell=n)))
        print("{0} took {1} secs to run".format(n, time()-t0))

    return cond_info

def _fix_cond_info(cond_info):
    '''
    change cond info from list of tuples of ndarrays to be ndarray.
    Then remove all the many identical tax keeping just one.
    Perform stats on fem and gating

    inputs:
    -------
        cond_info:      output of test_several_sigmoids
    '''
    tax = cond_info[0][0]

    fem = _np.zeros((len(cond_info), cond_info[0][0].shape[0]))
    gating = _np.zeros((len(cond_info), cond_info[0][0].shape[0]))

    for i, tup in enumerate(cond_info):
        fem[i,:] = tup[1]
        gating[i,:] = tup[3]

    nogating_mean = fem.mean(axis=0)
    nogating_std = fem.std(axis=0)
    gating_mean = gating.mean(axis=0)
    gating_std = gating.std(axis=0)

    fem = (fem, nogating_mean, nogating_std)
    gating = (gating, gating_mean, gating_std)
    
    return tax, fem , gating

def fake_noise(s_type, contrast, length=1000, mean=127):
    '''
    Fake a pink or gaussian stimulus depending on s_type

    input:
    ------
        s_type (str):   'pink' or 'gaussian'

        length:     in seconds

    output:
    -------
        stim (1D ndarray):        sequence of light intensities
    
    implementation notes:
        for the pink noise, I'm starting with a gaussian white noise -> rFFT -> dividing power in a freq by the freq -> iFFT
        The problem with this approach is that by changing the samples (time of experiment) I'm changing the lowest freq and so teh power goes to different freqs.
        I'm going to change it such that instead of generating the whole pink sequence at once, it is generated in chunks of about 5 secs
        
        I got code form someone on the web to generate the pink sequence. Seems to be working well until it doesn't. When computing cond_info between the LP and the noisy words I get some weird ripples that are clearly coming from the pink noise. Changing the noise amplitud to zero or using gaussian noise gets rid of the effect.

    '''
    #_pdb.set_trace()

    if contrast>=1:
        contrast/=100

    # each sample lasts sim_delta_t seconds in the simulation but to match the experiment, I don't want to flip the stimulus every sim_delta_t but rather every sim_delta_t*N, where N = number of samples in ~30ms
    monitor_flip_rate = .03
    N = int(monitor_flip_rate/sim_delta_t)
    #samples = int(samples/N)
    samples = int(length/(sim_delta_t*N))
    
    if s_type == 'gaussian':
        # grab random number with 0 mean and STD=1
        stim = _np.random.randn(samples)*mean*contrast + mean
    elif s_type == 'pink':
        stim = _pn.pink(samples)
        stim -= stim.mean()
        stim *= mean*contrast/stim.std()
        stim += mean
        """
        stim = _np.array([])
        samp_freq = 1/30    # this is the sampling freq of the monitor
        
        while stim.shape[0] < samples:
            next_samples = _np.random.randint(1/monitor_flip_rate, 10/monitor_flip_rate)
            next_stim = _np.random.randn(next_samples)
            stim_fft = _np.fft.rfft(next_stim)

            freq = _np.arange(1E-7, samp_freq/2+2E-7, samp_freq/samples)
            # smooth the power with a constant filter of lenght N
            #smooth_ker = _np.ones(N)/N
            #_np.correlate(power, smooth_ker, mode='valid')

            pink_fft = _np.divide(stim_fft, freq)

            next_stim = _np.fft.irfft(pink_fft)

            next_stim = next_stim*mean*contrast/next_stim.std()
            next_stim += mean - next_stim.mean()

            stim = _np.concatenate([stim, next_stim], axis=0)
        """
    else:
        raise ValueError('fake_noise requires s_type to be either "pink" or "gaussian"')

    # arange stim so that each intensity value lasts ~30ms as in the experiment. The simulation is set up such that each frame lasts sim_delta_t
    stim = stim.reshape(-1, 1) * _np.ones((1, N))
    stim = stim.reshape(-1, )
    
    return stim
   

def rate_increase(g, slope_range, thresh_range, binsN):
    '''
    compute the ratio of gating to nongating information for a bunch of conditions
    '''
    rate_increase_array = _np.zeros((len(slope_range), len(thresh_range)))

    for i, slope in enumerate(slope_range):
        for j, thresh in enumerate(thresh_range):
            nl = nonlinear_block('sigmoid', {'thresh':thresh, 'slope':slope})

            _, _, rate_increase_array[i, j] = compute_gating_effect(g, [.08, .1], nl, binsN)

    return rate_increase_array


def get_response(g, letter_times, letter_nl):
    '''
    Pass each g corresponding to letter_times through the corresponding letter_nl

    letter_times and letter_nl should have the same number of elements. 
    Letter i responses are:         letter_nl[i].torate(g[:, letter_time[i])

    input:
    ------
        g (2d ndarray):         linear prediction. g[i,j] corresponds to cell i, time point 'j' in the simulation

        letter_times:           iterable of N floats.
                                Corresponds to the times in g where letters will be extracted from

        letter_nl:              iterable of N nonlinear objects
                                Corresponds to the nonlinear objects that will be used to translate g[:, letter_time[i]] into a response.

    output:
    -------
        resp:                   2D ndarray with shape equal to (g.shape[0], len(letter_times))
    '''

    pass

                                
def compute_gating_effect(g, letter_times, nl, binsN):
    '''
    * bin g using binsN
    # get words by passing g at letter_times through nonlinearity. Last letter gets also passed through gating_nl
    * bin words using binsN
    * compute Shannon's I(last letter; g | previous letter)
    '''

    #_pdb.set_trace()
    time_points = [time_to_point(t, 0) for t in letter_times]
    
    # try binning g in an intelligent way, using percentiles. Each bin takes 100/2**binsN chuncks of data
    g_binned = _info.binned(g[:, time_points[-1]], binsN, 1)

    # pass linear prediction at the letters of interest through nonlinearity
    bin_rate = g.max()/binsN
    words = nl.torate(g[:, time_points], bin_rate = bin_rate)
    gating_letter = nl.gating_rate(g[:, time_points[-1]], bin_rate = bin_rate)
    
    if _np.isnan(max(words.flatten())) or _np.isnan(max(gating_letter)):
        raise ValueError('eihter words or gating_letter got "NaN"s inside compute_gating_effect')

    non_gating_info =  _info.cond_mi(words[:, -1], g_binned, words[:,0])
    gating_info = _info.cond_mi(gating_letter, g_binned, words[:,0])
    return gating_info, non_gating_info

def wrap_gating_effect(g, letter_length, nl, binsN):
    '''
    wrapper to call compute gating_effect with 2L words at all possible times

    outptu:
    -------
        save to file "Data/gating_effect_2L_{letter_length}ms" and "Data/nogating_effect_2L_{letter_length}ms"
    '''
    tax = _get_simulation_TAX()

    # allocate output
    gating = _np.zeros_like(tax)
    nogating = gating.copy()

    start_p = letter_length/sim_delta_t
    for p, t in enumerate(tax):
        if p < start_p:
            continue

        letter_times = [t-letter_length, t]
        gating[p], nogating[p] = compute_gating_effect(g, letter_times, nl, binsN)
        
    gating.tofile(datapath+'gating_effect_2L')
    nogating.tofile(datapath+'nogating_effect_2L')


def load_model_fit():
    '''
    In natural_scenes_fitting I loaded all PSTHs corresponding to a cell (all contrasts) and fitted a model where the threshold and peripheral_weight were variables.

    All results were stored in file: 'UFlicker PSTHs/best_parameters.txt' which is composed of 4 fields, the cell #, peripheral_weight, nl_threshold, and the error between the fit and teh PSTH
    
    Load that information and create and return a dictionary with cell # as key and a tuple with nonlinearity objects as values. Nonlinearities are of 'birect' form with just a threshold. Base nonlinearity uses nl_thresh as threshold and gated_nl has nl_thresh + peri_weight.

    dict[0] = (base_nonlinearity, gated_nonlinearity)
    '''
    df = _pd.read_csv('UFlicker PSTHs/best_parameters.txt', sep=' ')
    
    #_pdb.set_trace()
    nls = {}
    for i in df.index:
        base_nl = nonlinear_block('birect', df.iloc[i]['nl_thresh'], nl_units)
        gated_nl = nonlinear_block('birect', df.iloc[i]['nl_thresh']-df.iloc[i]['peri_weight'], nl_units)

        nls[df.iloc[i]['cell_id']] = (base_nl, gated_nl)

    return nls

def nls_to_list(nls, sele = None):
    '''
    After loading all model fits into nls, nls is a dictionary with cells as keys and tuples as values. Each tuple holds 2 nonlinearity objects. The first for basal condition and the second for gating.

    I want to get a list with either all the nonlinearities, or those from 'basal' or 'gating'

    '''

    #_pdb.set_trace()
    nl_list = []
    for nl in nls.values():
        if sele is None or sele is 'basal':
            nl_list.append(nl[0])
        if sele is None or sele is 'gating':
            nl_list.append(nl[1])

    return nl_list


def load_sigmoids(s_file = 'UFlicker_sigmoids.txt', length=100, contrast=3):
    '''
    Load all sigmoidal fits to gating cells from UFlicker experiment and convert them to nonlinear_block objects.
    '''

    dataframe = _load_sigmoids_dataframe(s_file=s_file, length=length, contrast=contrast)
    return _dataframe_to_nonlinear_block_list(dataframe)

def _load_sigmoids_dataframe(s_file = 'UFlicker_sigmoids.txt', length=None, contrast=3):
    '''
    Load all sigmoidal fits to gating cells from UFLicker experiment

    inputs:
    -------
        s_file:         plain txt file exported from igor with
                        day retina length cell contrast mask TW0_w[0] TW0_w[1] TW0_w[2] TW0_w[3] TW1_w[0] TW1_w[1] TW1_w[2] TW1_w[3] TW2_w[0] TW2_w[1] TW2_w[2] TW2_w[3] TW3_w[0] TW3_w[1] TW3_w[2] TW3_w[3]
        
        length:         length in seconds of the UFlicker experiment to use, I usually work only with 100s

        contrast:       which contrast to load

    output:
    -------
        newDF (dataframe):  Dataframe with parameters for all experimental nonlinearities
                            Either work directly with it or call _dataframe_to_nonlinear_block_list(newDF)
    '''

    # first load all sigmoids
    df = _pd.read_csv(s_file, sep=' ', parse_dates=['day'])

    # restrict df to those sigmas with contrast == 3 and mask==1 and no no null sd
    #df_3 = df[(df['contrast']==contrast) & (df['mask']==1) & (df['TW0_w[3]']!=0) & (df['TW1_w[3]']!=0) ]
    df_3 = df[(df['contrast']==contrast) ]

    if length is not None:
        df_3 = df_3[df_3['length']==length]

    # I have manually selected a bunch of gating cells to run the analysis on.
    # I will only keep those
    # not nice but effective
    newDF = _pd.DataFrame()
    for cell in [1,2,4,10,11,12,15,16,17,18,19,20,21]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='100928') & (df_3['retina']==1) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)

        if cell == 1:
            newDF['gatingTW'] = _pd.Series(1, index=newDF.index)
        # add a column with information on which TW to use for gating
        if cell == 1:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 2)
        else:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element

    for cell in [1,2,6,7,9,21,23,24]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='101011') & (df_3['retina']==2) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)
        # add a column with information on which TW to use for gating
        if cell == 1:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 2)     # change only last element
        elif cell == 6:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 3)     # change only last element
        else:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element

    for cell in [1,6,7,12,14,15,18,21]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='101206') & (df_3['retina']==1) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)
        # add a column with information on which TW to use for gating
        if cell == 1:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 3)     # change only last element
        elif cell in [15,21]:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element
        else:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 2)     # change only last element
    
    for cell in [1,2,3,6]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='110106') & (df_3['retina']==2) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)
        # add a column with information on which TW to use for gating
        if cell == 2:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 2)     # change only last element
        else:
            newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element
        
    for cell in [1,5,8,9,35]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='110204') & (df_3['retina']==1) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)
        newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element
    
    for cell in [1,4,7,10,17,18,20,22]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='110420') & (df_3['retina']==2) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)
        newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element

    for cell in [1,2,3,4,6,7,8]:
        newDF = _pd.concat([newDF, df_3[(df_3['day']=='110516') & (df_3['retina']==1) & (df_3['cell'] == 'c{0}'.format(cell))]], ignore_index=True)
        newDF.set_value(len(newDF.index)-1, 'gatingTW', 1)     # change only last element

    #newDF.append((df_3['day']==100928) & (df_3['retina']==1) & (df_3['cell'] in [2,4,10,11,12,15,16,17,18,19,20,21]))
    
    newDF.reset_index()

    return newDF

def _dataframe_to_nonlinear_block_list(df):
    '''
    convert data frame with all nonlinearities to a list of nonlinear_block objects
    '''
    sigmoids = []

    #_pdb.set_trace()
    # loop through the data frame and for each line, create a nonlinear_block object that has TW0 as the base nonlinearity and the one pointed at by gatingTW as the gating one.
    for i in df.index:
        row = df.iloc[i]

        # original nonlinearities are in contrast units (contrast in the 0-1 range). Remove that, dividing both threshold and sd by the contrast
        sigmoid = nonlinear_block('sigmoid', row['TW0_w[2]'], 'linear prediction', contrast = row['contrast'], min_fr = row['TW0_w[0]'], max_fr = row['TW0_w[1]']+row['TW0_w[0]'], sd=row['TW0_w[3]'])
        sigmoid.gating_nl = nonlinear_block('sigmoid', row['TW{0}_w[2]'.format(int(row['gatingTW']))], 'linear prediction', contrast = row['contrast'], 
                min_fr = row['TW{0}_w[0]'.format(int(row['gatingTW']))], 
                max_fr = row['TW{0}_w[1]'.format(int(row['gatingTW']))] + row['TW{0}_w[0]'.format(int(row['gatingTW']))], 
                sd = row['TW{0}_w[3]'.format(int(row['gatingTW']))])

        sigmoids.append(sigmoid)

    return sigmoids

def explore_word_letters(g, start_t, end_t, bipolar, bin_rate):
    '''
    plot a histogram of letters during the start_t, end_t under both gating and no gating nonlinearities
    '''
    #_pdb.set_trace()
    start_p = time_to_point(start_t, 0)
    end_p = time_to_point(end_t, 0)

    # limit g to the gating window
    g_temp = g[:, start_p:end_p]
    
    # pass g through both non gating and gating nl
    nongating_letters = bipolar.nl_basal.torate(g_temp, bin_rate=bin_rate)
    gating_letters = bipolar.nl_gating.torate(g_temp, bin_rate=bin_rate)

    # plot distribution of letters with gating and no gating nls during the gating window
    _plt.close('letter_distribution')
    _plt.figure('letter_distribution')
    _plt.hist([gating_letters.flatten(), nongating_letters.flatten()],bins=10, cumulative=True)

def fit_all_TNF_cells():
    '''
    load all PSTHs for TNF experiment and fit the best peripheral_weight and nl_threshol that replicates the data.
    Store all those parameters in TNF_PSTH_fits.txt
    '''
    #_pdb.set_trace()
    bipolar = cell(letter_length_in_ms)

    still_psths, sac_psths, tax = load_TNF_PSTHs()

    fid = open('TNF_PSTH_fits.txt', 'wt')
    fid.write('peri_weight nl_thresh avg_fr error\n')

    for i in range(0,sac_psths.shape[0]):
        if i in [49, 85, 118]:
            fid.write('NaN NaN NaN NaN\n')
            continue

        print('Fitting cell {0}'.format(i, sac_psths.shape[0]))
        best_params = bipolar._fit_PSTH(sac_psths[i,:], 'pink', .1, 127, 1000, 96, range(-10, 200, 10), range(-100, 100, 10))
        fid.write('{0} {1} {2} {3}\n'.format(best_params[0], best_params[1], best_params[2], best_params[3]))

        print('\t\t {0}'.format(best_params[0], best_params[1]))
    fid.close()

def _save_TNF_PSTHs_for_selection(sac_psths):
    '''
    load file TNF_PSTH_fits.txt (generated with fit_all_TNF_cells) and display and save the PSTH along side the best fit.
    
    '''

    for i in range(sac_psths.shape[0]):
        _plt.close('TNF_fits')
        fig, ax = _plt.subplots(num='TNF_fits')

        ax.plot(sac_psths[i,:])
        fig.savefig('TNF PSTHs/c{0}'.format(i))


def plot_stats_from_TNF_fits(g):
    '''
    load file TNF_PSTH_fits.txt (generated with fit_all_TNF_cells) and compute the average and std of peri_weigth and nl_thresh
    
    g:      linear prediction, being used to change scale of thresholds from linear predictions to std of g during FEM

    output:
    -------
        peri_weight.mean(), peri_weight_std(), nl_thresh.mean(), nl_thresh.std() 
    '''
    #_pdb.set_trace()
    
    df = _pd.read_csv('TNF_PSTH_fits.txt', sep=' ')

    rows_to_drop = [13, 15, 23, 32, 34, 36, 47, 48, 49, 50, 53, 54, 55, 56, 57,58, 61, 62, 73, 85, 90,98, 99, 100, 101, 102, 103,105, 107, 109, 118, 124, 126, 128]
    df = df.drop(df.index[rows_to_drop])

    _plt.close('TNF_fit_stats')
    fig, ax = _plt.subplots(nrows=1, num='TNF_fit_stats')

    ax.hist(df['peri_weight'], bins=50, color='k', histtype='stepfilled', normed=True, alpha=.5)
    #ax[1].hist(df['nl_thresh'], bins=50, color='k', histtype='stepfilled', normed=True, alpha=.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax[1].spines['top'].set_visible(False)
    #ax[1].spines['right'].set_visible(False)

    ax.tick_params(length=3, right='off', top='off')

    # X axis, instead of being in linear prediction units will be relative to STD during FEM. 1) Compute std during FEM
    FEM_points = (-.005 - sim_start_t)/sim_delta_t
    std = g[:, :FEM_points].flatten().std()

    xticks = _np.arange(0, 3.1*std, std)
    ax.set_xticks(xticks)
    ax.set_xticklabels(range(len(xticks)+1), fontsize=10)
    ax.set_xlabel(r'$Threshold\,Shift\, (FEM\, std)$', fontsize=12)
    ax.set_xlim(0, 3*std)

    yticks = _np.arange(0, .05, .025)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=10)
    ax.set_ylabel(r'$probability$')
    
    fig.subplots_adjust(left=.2, bottom=.3, right=.95, top=1)
    fig.set_size_inches(2.5, 2)
    fig.savefig('Figures/TNF_fit_stats.pdf', transparent=True)
    return fig, df['peri_weight'].mean()/std, df['peri_weight'].std()/std, df['nl_thresh'].mean()/std, df['nl_thresh'].std()/std 

def compute_kernel(stim, resp, pnts_before_spk, pnts_after_spk):
    '''
    given a linear prediction and responses, compute the filter

    pnts:   final number of pints in filter

    start_t:        seconds before spike to include in kernel (should be <0)

    '''
    #_pdb.set_trace()
    assert stim.shape == resp.shape, 'naturalscenes.compute_kernel: lp and resp should have the same shape'

    kernel = _np.zeros(pnts_before_spk + pnts_after_spk )

    
    stim -= stim.mean()
    for i in range(pnts_before_spk, len(stim)-pnts_after_spk):
        kernel += resp[i] * stim[i-pnts_before_spk:i+pnts_after_spk]

    if kernel.max() != kernel.min():
        kernel = kernel/_np.sqrt(_np.dot(kernel,kernel))
    
    return kernel[::-1]

def adaptive_index(slopes, stds):
    '''
    Compute adaptive index
    
    From a set of nonlinearities corresponding to Gaussian stimuli of different contrasts compute the slopes.
    Normalize slopes to the largest one in the set. 
    Fit a line to 'normalized slopes' vs 1/sigma_i (the standard deviation of the gaussian contrast)
    Adaptive index is the slope of the fitted line
    
    input:
    -----
        slopes:     array with the gain of each nonlinearity under the corresponding Gaussian distribution

        stds:       std of the gaussian distributions that generated nonlinearities with the given slopes
    
    output:
    ------
        AI:         adaptive index
    '''

    slopes/=slopes.max()

    inverse_std = _np.divide(1, stds)

    return _np.polyfit(inverse_std, slopes, 1)[0]

def simulate_adaptive_index():
    '''
    Simulate UFlicker under several contrasts.
    For each contrast compute the nonlinearities at different TWs
    Fit nonlinearities with a birect curve, and extract the slope of the 2nd one
    With the slopes and the STDs of the Gaussians, compute adaptive_index in each TW
    Plot adaptive index as a funciton of time window
    '''
    #_pdb.set_trace()
    contrasts = [3,6,12,24,48]
    contrasts = [6,12,24,48,96]
    bipolar = cell(.005)
    mean = 127

    TWN = 12
    ai = _np.zeros(TWN)
    slopes = _np.zeros((len(contrasts), TWN))
    for i, contrast in enumerate(contrasts):
        _, _, nls = bipolar.simulate_UFlicker(mean, contrast, 200, peri_factor = 1, TWN=TWN, plot_flag=0)
        print("contrast is ", contrast)
        
        #fig = _plt.gcf()
        #ax = fig.get_axes()
        for j, nl in enumerate(nls):
            print("nl is ", j)
            _, _, line_1, fit, _ = fit_birect(nl[0], nl[1])
            #ax[12+j].plot(nl[0], fit)
            slopes[i,j] = line_1[0]

    for i in range(TWN):
        ai[i] = adaptive_index(slopes[:,i], contrasts)

    _plt.close('adaptation_index')
    fig, ax = _plt.subplots(num='adaptation_index')
    ax.plot(ai)
    

    plot_adaptive_index_slopes(contrasts, slopes, [0,2,5])
    
    return ai, slopes, contrasts

def fit_birect(nl_x, nl_y):
    '''
    Given a non-linearity described by nl_y vs nl_x, fit it with two lines.

    input:
    -----
        nl_x/y      1d arrays with the same number of points, plot(nl_x, nl_y) should give the nonlinearity

    output:
    ------
        best_p:     1st line accounts for nl_x[:best_p], nl_y[:best_p]
                    2nd line accounts for nl_x[best_p:], nl_y[best_p:]

        line0/1:    output of polyfit for each of the two lines      

        best_fit:   1d array with same number of points as nl_x/y with the combination of both lines.
                    used for: plot(nl_x, best_fit)

        error:      sum of square errors
    '''
    #_pdb.set_trace()

    N = len(nl_x)
    error = _np.infty

    # loop through all points in nl_x/y, dividing the data in two parts and fit a line to each one. Keep track of total error and return best two fits
    if nl_y.max()==0:
        best_p = _np.nan
        line0 = (0,0)
        line1 = (0,0)
        best_fit = nl_y
        error = 0
    else:
        for i in range(3,N-3):
            #sim_noise_fit = _np.poly1d(_np.polyfit(df['sim_mp_sd'], df['sim_mp_noise'],1))
            if nl_y[:i].max() == 0:
                res0 = [0]
                fit0 = (0,0)
            else:
                fit0, res0, _, _, _ = _np.polyfit(nl_x[:i], nl_y[:i], 1, full=True)

            if nl_y[i:].max() == 0:
                res1 = [0]
                fit1 = (0,0)
            else:
                fit1, res1, _, _, _ = _np.polyfit(nl_x[i:], nl_y[i:], 1, full=True)
            
            if res0[0]*(N-i) + res1[0]*N < error:
                error = res0[0]*(N-i) + res1[0]*N
                line0 = fit0
                line1 = fit1
                best_p = i

        # make an array to plot on top of nl_y
        best_fit = line0[0]*nl_x + line0[1]
        best_fit[best_p:] = line1[0]*nl_x[best_p:] + line1[1]
    
    return best_p, line0, line1, best_fit, error

def compute_nl(lp, resp, pnts):
    '''
    given a linear prediction and responses (both 1d ndarrays of the same length), compute the nonlinearity

    inputs:
    -------
        lp:     linear prediction, output of convolving stim with filter

        resp:   cells response, can be spikes or membrane potential but has to be alligned with lp (same number of points, etc)

        pnts:   final number of points in the nonlinearity

    output:
    -------
        lp_short:       x axis of nonliearity with 'pnts'

        resp_short:     y axis of nonliearity with 'pnts'

        sorted_lp:      x axis of nonlinearity with as many points as original lp

        sorted_resp:    y axis of nonlinearity with as many points as original resp

    '''

    assert lp.shape == resp.shape, 'naturalscenes.compute_nl: lp and resp should have the same shape'

    # get the indexes that would sort lp
    indexes = lp.argsort()

    # sort lp and resp according to that index but not in place
    sorted_lp = lp[indexes]

    sorted_resp = resp[indexes]

    down_sample_n = int(len(lp)/pnts)
    lp_short = average(sorted_lp, down_sample_n, 4)
    resp_short = average(sorted_resp, down_sample_n, 4)

    return lp_short, resp_short, sorted_lp, sorted_resp


def correlations_cartoon(bits_info, bits_noise):
    '''
    Explain with a cartoon example why we need conditional information

    A word (w) is defined as a vector of letters w = (L0, L1, ..., L(n-1))
    '''
    stim, noisy = _fake_correlated_stim(bits_info, bits_noise, 1000)

    # Info with 1 Letter word
    info_0 = mi(stim,stim)
    noisy_0 = mi(stim, noisy)

    # now compute the information that the perfectly linear encoder with 2 letters conveys about the correlated stim with the 2nd letter (newest)
    info_L0 = mi(stim[1:], stim[:-1])
    info_L1 = cond_mi(stim[1:], stim[1:], stim[:-1])

    # Information of a 2-letter word about stim when noise is added
    noisy_L0 = mi(stim[1:], noisy[:-1])
    noisy_L1 = cond_mi(stim[1:], noisy[1:], noisy[:-1])

    # Information with a 3-letter word about the stim when noise is added
    noisy_3L_L0 = mi(stim[2:], noisy[:-2])
    noisy_3L_L1 = cond_mi(stim[2:], noisy[1:-1], noisy[:-2])
    noisy_3L_L2 = cond_mi(stim[2:], noisy[2:], combine_labels(noisy[:-2], noisy[1:-1]))

    # Information with 1L with no noise but 2 frames
    info_F0 = mi(stim[:-1], stim[1:])
    info_F1 = cond_mi(stim[1:], stim[1:], stim[:-1])
    noisy_F0 = mi(stim[:-1], noisy[1:])
    noisy_F1 = cond_mi(stim[1:], noisy[1:], stim[:-1])

    # Information with a 3-frame stimulus and 1-L word when noise is added
    noisy_3F_F0 = mi(stim[:-2], noisy[2:])
    noisy_3F_F1 = cond_mi(stim[1:-1], noisy[2:], stim[:-2])
    noisy_3F_F2 = cond_mi(stim[2:], noisy[2:], combine_labels(stim[:-2], stim[1:-1]))

    _plt.close('correlations_cartoon')
    fig, ax = _plt.subplots(nrows=4, ncols=2, num='correlations_cartoon', sharex=True, sharey=True)

    tax = _np.arange(sim_start_t, sim_end_t, sim_delta_t)


    ax[0][0].plot(tax, info_0)
    ax[0][0].plot(tax, noisy_0)
    ax[0][0].plot([sim_start_t, -sim_delta_t,0,sim_end_t], [0, 0, 1, 1], "k")

    ax[1][0].plot(tax[1:], info_L0)
    ax[1][0].plot(tax[1:], info_L1)
    ax[1][0].plot(tax[1:], info_L0+info_L1, ":k")

    ax[2][0].plot(tax[1:], noisy_L0)
    ax[2][0].plot(tax[1:], noisy_L1)
    ax[2][0].plot(tax[1:], noisy_L0+noisy_L1, ":k")

    ax[3][0].plot(tax[2:], noisy_3L_L0)
    ax[3][0].plot(tax[2:], noisy_3L_L1)
    ax[3][0].plot(tax[2:], noisy_3L_L2)
    ax[3][0].plot(tax[2:], noisy_3L_L0+noisy_3L_L1+noisy_3L_L2, ":k")

    ax[1][1].plot(tax[1:], info_F0)
    ax[1][1].plot(tax[1:], info_F1)
    ax[1][1].plot(tax[1:], info_F0+info_F1, ":k")

    ax[2][1].plot(tax[1:], noisy_F0)
    ax[2][1].plot(tax[1:], noisy_F1)
    ax[2][1].plot(tax[1:], noisy_F0+noisy_F1, ":k")

    ax[3][1].plot(tax[2:], noisy_3F_F0)
    ax[3][1].plot(tax[2:], noisy_3F_F1)
    ax[3][1].plot(tax[2:], noisy_3F_F2)
    ax[3][1].plot(tax[2:], noisy_3F_F0+noisy_3F_F1+noisy_3F_F2, ":k")

    # make plot prettier
    ax[0][0].set_ylim(-1, bits_info+1)
    ax[0][0].set_xlim(-.2, .8)
    xticks = _np.arange(-.2, .8, .2)
    ax[0][0].set_xticks(xticks)
    yticks = [0,1,4]
    ax[0][0].set_yticks(yticks)

def _fake_correlated_stim(stim_bits, noise_bits, trials):
    '''
    fake stim and noisy versions.
    
    stim follows a uniform distribution with 'stim_bits' information and is correlated over time.

    noisy, is stim + a uniform distribution with 'noise_bits' of information and no correlations.

    
    input:
    -----
        stim_bits:  total uncertainty in the stimulus conditioning on a change of stimulus
                    when the stimulus changes, the new value is picked out of 2**bits possibilities

        noise_bits: uncertainty in the noise at every frame

        trials: number of saccades in the stimulus

    output:
    ------
        noise:  1d array
    '''
    #_pdb.set_trace()

    #samples per saccade
    frames_per_saccade = (sim_end_t-sim_start_t)/sim_delta_t

    # since the stim changes every correlation_length I only need stim_length/correlation_length different values
    # I'm generating it to be already in such a way that stim[i,j] represents time i, cell j which is what I am using in informaiton calculations
    stim = _np.random.random_integers(0, 2**stim_bits-1, (trials,1))*_np.ones((1,frames_per_saccade))
    #stim = _np.ones((frames_per_saccade,1))*_np.random.random_integers(0, 2**bits-1, (1,trials))


    # shift stim such that saccades happen at time 0
    shift = int(-sim_start_t/sim_delta_t)
    stim = _np.roll(stim, shift)

    noisy = stim + _np.random.random_integers(0, 2**noise_bits-1, (trials,frames_per_saccade))
    
    
    # return tuple version stim and noise, where stim[i][j] is time i, cell j
    stim = tuple(map(tuple, stim.T))
    noisy = tuple(map(tuple, noisy.T))
    
    return stim, noisy


def _test_adaptation(contrast_list, filter_instance, nl, adaptation_block):
    _plt.close('adaptation_test')
    fig, ax = _plt.subplots(nrows=3, num='adaptation_test')
    
    #_pdb.set_trace()

    gauss=[]
    ca_concentration=[]
    adapted_output=[]
    labels=[]
    for c in contrast_list:
        # filter some gaussian noise, pass it through a nl and adapt the output
        gauss.append(filter_gaussian_noise(filter_instance, c, samples=10000))

        # In order to get adaptation under "memory_normalization", the "memory" of the signal has to scale with its variance.
        # If for example there is a threshold nonlinearity before adaptation block but the threshold is too low, such that all the signal is in the linear range, then the mean averaged by the memory will be independent of the signal's variance and there will be no effective normalization.
        # The most effective normalization happens when the threshold is right in the middle of the distribution such that half the values are cliped and the mean goes like the standard deviation of the signal
        nl.thresh = gauss[-1].mean()

        ca_concentration.append(nl.torate(gauss[-1]))

        adapted_output.append(adaptation_block.adapt(ca_concentration[-1]))

        labels.append('C={0}%'.format(c))

    bins = 50
    ax[0].hist(gauss, bins=bins, normed=True, alpha=1, label=labels)

    ax[1].hist(ca_concentration, bins=bins, normed=True, alpha=1)

    #return adapted_output
    ax[2].hist(adapted_output, bins=bins, normed=True, alpha=1)

    ax[0].legend()

    return adapted_output



class nonlinear_block:
    def __init__(self, s_type, thresh, units, contrast=None, min_fr=0, max_fr=1, sd=1, slope=1):
        '''
        Init a nonlinear_block object
        Not all imput parameters are used, depending on your choice of 's_type'

        inputs:
            most are self explanatory but...
            
            s_type:         'birect',   usses thresh and slope
                            'sigmoid',  uses min_fr, max_fr, sd, slope

            units:          'linear prediction' or 'sd of linear prediction'
                            if units is 'sd of linear prediction' then before passing a signal through NL the signal has to have SD == 1
                            if units is 'linear prediction' then signal passing through NL can have any SD

        '''
        # s_type can either be 'sigmoid' or 'birect'
        if s_type not in ['sigmoid', 'birect']:
            raise ValueError('s_type has to be either "sigmoid" or "birect"')

        if contrast is not None and contrast>1:
            contrast /=100.0 

        if units not in ['linear prediction' or 'sd of linear prediction']:
            raise ValueError('units has to be either "linear prediction" or "sd of linear prediction"')

        #_pdb.set_trace()
        self.s_type = s_type
        self.units = units
        self.thresh = thresh
        self.contrast = contrast
        self.min_fr = min_fr
        self.max_fr = max_fr
        self.sd  = sd
        self.slope = slope

    def copy(self):
        return nonlinear_block(self.s_type, self.thresh, self.units, contrast=self.contrast, min_fr=self.min_fr, max_fr=self.max_fr, sd=self.sd, slope=self.slope)

    def torate(self, linear_prediction, bin_rate=None):
        '''
        pass the linear prediction through the nonlinerity

        units:  0 input is in contrast units. Under ideal adaptation measured nonlinearities at different contrast should overlay each other
                1 input is in light units. Measured nonlinearities at different contrast will not overlay each other.
        
        bin_rate:   if bin_rate is given, responses are discretized by floor(rate/bin_rate)
        '''

        #_pdb.set_trace()

        # store original shape since map works on 1D objects and I will flatten the linear_prediction
        shape_ori = linear_prediction.shape


        # preallocate firing_rate ndarray
        firing_rate = _np.zeros_like(linear_prediction)

        if self.units=='sd of linear prediction':
            raise ValueError('not well implemented.')
            # it seems to me that both the filter, nl and linear prediction should have a 'units' property that they check before interacting with each other.
            thresh = self.thresh*1.0/self.contrast
            slope = self.slope*1.0/self.contrast
            sd = self.sd*1.0/self.contrast
        else:
            thresh = self.thresh
            slope = self.slope
            sd = self.sd

        if sd==0 or slope==0:
            return firing_rate

        #_pdb.set_trace()
        if self.s_type == 'sigmoid':
            from scipy.special import expit
            # pass g through the nonlinearity. I'm using scipy.special.expit which is extremely fast, but requires changing the input according to threshold and sigma
            firing_rate = expit((linear_prediction-thresh)/sd)*self.max_fr + self.min_fr
        elif self.s_type == 'birect':
            '''
            lp1d = linear_prediction.flatten()
            firing_rate = _np.array(list(map(lambda x: 0 if x < thresh else slope*(x - thresh), lp1d))).reshape(shape_ori)
            '''
            firing_rate = (linear_prediction-thresh)*slope
            below_thresh_indices = firing_rate<0
            firing_rate[below_thresh_indices] = 0

        if bin_rate is not None:
            firing_rate = _np.ceil(firing_rate/bin_rate)

        return firing_rate

    def __copy__(self):
        return nonlinear_block(self.s_type, self.thresh, self.units, self.contrast, self.min_fr, self.max_fr, self.sd, self.slope)

    
class adaptation_block:
    '''
    Define an adaptive block, for the time being I'm only implementing dividing by memory+offset
    '''

    def __init__(self, s_type, memory, delta_t, offset = 0):
        '''
        Input:
        ------
            s_type:     for the time being only "memory_normalization"

            memory:     float, in seconds

            detla_t:    time resolution of array to be adapted

            offset:     float, in the same units as the linear prediction

        '''
        from numpy.linalg import norm

        self.s_type = "memory_noramlization"
        self.memory = memory
        self.delta_t = delta_t
        self.offset = offset


        #_pdb.set_trace()
        # convolve signal with an exponential that decays to 1/e over "memory" seconds
        # time unit in the exponential is the same as in signal, letter_length (in seconds)
        # I'm making the exponential to be long enough such that the last point contributes .01 times what the first point contributes exp(-last/memory)=0.01
        # last = -memory_p*log(0.01)        (is in seconds)
        last = -self.memory*_np.log(.01)
        self.memory_array = _np.exp(-1*_np.arange(0, last, delta_t)/self.memory)
        
        # make memory_array such that the sum of its elements is 1 (convolve(ones(1000), memory_array) still gives ones)
        self.memory_array /= self.memory_array.sum()

        # before convolving make memory_array of unit norm
        #self.memory_array /= norm(self.memory_array)

    def adapt(self, signal, del_flag=0):
        '''
        signal is probably going to be [Ca]. Divide signal by the result of convolving signal with a decaying exponential with 'memory' and adding an offset

        signal can be 1d array or nd array. Last dimension is the time dimension to be convolved by the decaying exponential

        by convolving, 
        '''

        #_pdb.set_trace()
        N = len(self.memory_array)
        # convolve signal with memory_array. I work on a flatten version of array
        convolution = _np.convolve(signal.flatten(), self.memory_array)

        # now convolution has len(memory_array) + len(signal.flatten()) points. I'm discarding points from the end to keep convolution the same size as signal.
        # Then I'm reshaping it to be the original size. 
        convolution = convolution[:- N +1].reshape(signal.shape)

        if del_flag:
            # From every row now, the first len(memory_array)-1 points are trash because the filter is overlaping signal from different cells. After reshaping to signal's original shape, remove those columns
            convolution = _np.delete(convolution, range(N-1), convolution.ndim-1)

            signal = _np.delete(signal, range(N-1), signal.ndim-1)

        # divide signal by convolution + offset. Convolution is longer than signal by len(memory_array)-1 points that were added to the front. That's why in the following line I have convolution[len(memory_array)-1:]
        return _np.divide(signal, convolution + self.offset)


class filter_block:
    def __init__(self, size, kernel, weight, normed=True):
        '''
        Each filter block represents a decomposable space and time filter.
        For the time being, space is defined as a circular disk and images are filtered with it. If the disk size is 0, no filtering takes place and I will use it for Uniform stimulation. Time is defined through the kernel.

        By combining two of such filters I can accomplish the original simulation where center was one pathway and surround was another pathway, each space-time decomposable that where latter summed.
        I can also add a third pathway that is the peripheral one, with no spatial filter

        kernel can be either a 1D ndarray or a path to a file
        '''
        
        if isinstance(kernel, str):
            self.kernel = _np.fromfile(kernel, sep=' ')
        elif isinstance(kernel, _np.ndarray):
            self.kernel = kernel
        else:
            raise TypeError("naturalscenes.filter_block.__init__: kernel was not understood")

        self.size   = size
        self.weight = weight
        self._define_spatial_filter(size*pixperdegree)

    def filter_image(self, image):
        '''
        filter spatial image 'image' with a disk of size self.size
        
        if self.size == 0, no spatial filtering is done and image is returned.

        inputs:
        -------
            image:   2D ndarray with light intensities (probably in the 0-255 range)

        outputs:
        --------
            self.filtered:       filtered image with self.size disk
        '''
        #_pdb.set_trace()
        self.filtered_image = _nd.uniform_filter(image, self.size * pixperdegree, mode='constant')

    def temporal_filter(self, stim):
        '''
        simulate the membrane potential of a cell centered on center = (centerX, centerY) moving according to seq
        
        inputs:
        -------
            stim:       Temporal values of the stimulus prior to kernel filtering.
                        Stim can be the output of spatialy filtering an image and moving it around according to eye movements to generate a temporal stimulus
                        Stim can also be a sequence of intensities as in uniform flickering where no spatial integration takes place.
        
                        for example if self.filter_image(some_image) was called, generating self.filtered_image and a sequence of eye movements 'eye_seq' exists, the following extract the temporal stimulus as seen by a cell centered at 'center' = (centerX, centerY).
                        
                        stim = _np.array([self.filtered_image[seq[0,i]+center[0]][seq[1,i]+center[1]] for i in range(seq.shape[1])])
                        
                        where seq is a 2D ndarray sequence of positions relative to centerX, centerY
                        seq[0][:] are positions in x
                        seq[1][:] are positions in y

        output:
        -------
            mp:         the membrane potential contribution coming out of this pathway
                        mp will be shorter than stim. In particular, to allign mp with stim or any other array like stim the 1st len(kernel)-1 points from stim should be removed
                        for example if kernel = array(1,0,0,0,0) then mp = stim[4:]
                        see test_naturalscenes.test_temporal_filter
        '''

        # Filter the center and the surround by its corresponding kernel
        mp = self.weight * _np.convolve(stim, self.kernel, mode='valid')
        
        # combine center and surround
        return mp

    def mean_adapt(image):
        '''
        simulate mean adaptation by forcing image mean to be 127
        '''
        # simulate light adaptation, change image mean to be 127.5
        im *= 127.5/im.mean()

        return image

    def contrast_adapt(image):
        '''
        simulate contrast adaptation by forcing image to be in 0-255 range
        '''
        image = 2**8 * (image-image.min())/(image.max() - image.min())

    def _define_spatial_filter(self, size_in_pixels):
        '''
        Define the spatial filter to be used
        '''

        # make sure that size_in_pixels is int and odd
        size_in_pixels = int(size_in_pixels)

        if size_in_pixels/2==0:
            size_in_pixels+=1

        # create a 2D array of size_in_pixels x size_in_pixels with the weight that each pixel contributes to filtered image.
        half = _np.ceil(size_in_pixels/2)
        y, x = _np.ogrid[-half:half+1, -half:half+1]
        self.spatial_filter = y**2 + x**2 < half**2
        self.spatial_filter = self.spatial_filter / self.spatial_filter.sum()

class cell:
    def __init__(self, letter_length_in_ms):
        '''
        letter_length is in seconds and is used to generate memory kernel in adaptation_block
        '''

        #_pdb.set_trace()

        # define all pathways needed
        self.center = filter_block(center_size, center_kernel_file, center_weight)

        self.surround = filter_block(surround_size, surround_kernel_file, surround_weight)

        
        periphery_kernel = generate_peripheral_kernel(gating_start_t, gating_end_t, len(self.center.kernel), save_flag=1, display_flag=0)
        self.periphery = filter_block(periphery_size, periphery_kernel, periphery_weight, normed=False) #TODO how is the size of periphery input defined? SOme redundant parameters. Clean it up 

        # define noise model
        self.noise_model = self._get_mp_noise_model()

        # Define internal threshold
        self.nl_basal = nonlinear_block(nl_type, nl_basal_threshold, nl_units)
        self.nl_gating = nonlinear_block(nl_type, -nl_gating_amplitud, nl_units)
        self.nl_inh = nonlinear_block(nl_type, nl_gating_amplitud, nl_units)

        self.adaptation = adaptation_block(adaptation_type, adaptation_memory, letter_length_in_ms/1000, adaptation_offset)


    def processAllImages(self, maxImages=None, maxCellsPerImage=None):
        '''
        Compute the linear prediction of this cell as defined by parameters in self.center, self.surround, when the cell moves over many images from Tkacik's data base
        Cells are moving according to a FEM + a saccade that happens at time 0.
        Time in the simulation is defined by sim_start_t, sim_end_t and sim_delta_t, the time axis is tax = arange(sim_start_t, sim_end_t, sim_delta_t)

        inputs:
        -------
            maxImages:          integer, optional parameter defining the maximum number of images to use
                                defaults to None, meaning use all images
        
        outpus:
        -------
            g:                  2D ndarray, the linear prediction of many identical cells over many images
                                g[i][:]     is the linear prediction of cell i over time
                                g[:][t0]    is the linear prediction of all cells and all images at time t0
        '''
        #_pdb.set_trace()

        # try loading 'linear_prediction' from Data/, if that fails, compute it
        from os import listdir
        if 'linear_prediction' in listdir():
            sacc_path = 'Sacc={0}, RWS=.0{1}/'.format(saccade_size, int(rw_step*100))
            g = _np.fromfile(sacc_path + 'linear_prediction').reshape(-1,300)
            return g

        if images_list is None:
            _getImagesPath()
        

        # estimate number of cells per image
        if maxCellsPerImage is None:
            centerD = self.center.size*pixperdegree
            imSize = _loadImage(0).shape
            maxCellsPerImage = _np.floor(imSize[0]/centerD)*_np.floor(imSize[1]/centerD)

        # compute time axis of simulation
        tax = _get_simulation_TAX()

        # preallocate array for all linear predictions
        g = _np.zeros((maxCellsPerImage*len(images_list), len(tax)))
    
        #_pdb.set_trace()
        nextCell = 0
        for imNumber in range(len(images_list)):
            if imNumber == maxImages:
                break
            
            print(images_list[imNumber])
            t = _time()
            nextCell = self._processOneImage(imNumber, g, nextCell, maxCellsPerImage)
            print('\t{0} cells processed in {1} secs'.format(nextCell, _time()-t))

        g = g[:nextCell][:]
        g.tofile('linear_prediction')

        return g


    def _processOneImage(self, imNumber, g, nextCell, maxCells=None):
        '''
        Compute the linear prediction of several instances of these cell moving over the image described by imNumber

        inputs:
        -------
            imNumber:   integer, image to load from images_list
        
            g:          2D array with all inear predictions. Will be modified in place

            nextCell:   index into the 1st dimension of g where next simulated cell should be incorporated.
                        
            maxCells:   int, optional. If given limits how many cells will be processed on a given image.

        output:
            g:          modified in place, incorporates the linear predictions from image imNumber in g, starting from 
                        row = nextCell
        '''

        
        # filter image with center and surround spatial filters. Property 'filtered_image' is set in each filter_block
        self.filter_image(imNumber)

        # grab the eye movement sequence
        seq = _getEyeSeq(len(self.center.kernel))

        # grab non overlapping cells from image such that when moved according to seq, they are always whithing the boundaries
        centerD     = int(self.center.size * pixperdegree)    # center's diameter in pixels
        surroundD   = int(self.surround.size * pixperdegree)     # surround's diameter in pixels

        #_pdb.set_trace()
        image_size = self.center.filtered_image.shape

        startX  = int(_np.ceil(surroundD - min(seq[0][:])))
        endX    = int(_np.floor(image_size[0] - surroundD - max(seq[0][:])))
        startY  = int(_np.ceil(surroundD - min(seq[1][:])))
        endY    = int(_np.floor(image_size[1] - surroundD - max(seq[1][:])))
        
        i = 0
        #_pdb.set_trace()
        for center in _product(range(startX, endX, centerD), range(startY, endY, centerD)):
            # extract from filtered versions of image the time series corresponding to central and surround contributions of this particular cell
            center_stim  = _np.array([self.center.filtered_image[seq[0,i]+center[0]][seq[1,i]+center[1]] for i in range(seq.shape[1])])
            surround_stim = _np.array([self.surround.filtered_image[seq[0,i]+center[0]][seq[1,i]+center[1]] for i in range(seq.shape[1])])

            # pass those time series through the temporal filter and combine them
            g[nextCell, :] = self.center.temporal_filter(center_stim) + self.surround.temporal_filter(surround_stim)
            
            nextCell += 1
            i+=1
            if i==maxCells:
                return nextCell

        return nextCell


    def filter_image(self, imNumber):
        '''
        load given image and pass it through the spatial component of the filters
        '''
        #_pdb.set_trace()

        image = _loadImage(imNumber)

        self.center.filter_image(image)

        self.surround.filter_image(image)

    def _get_mp_noise_model(self, plot_flag=0):
        '''
        Simulate the same experiment that Yusuf did.
        Yusuf's experiment computes two different STDs under different gaussian contrast stimulation:
            1. the STD of the membrane potential over time (exp_mp_std, no repeats needed for this)
            2. the STD across responses of repeats of the same stimulus (exp_mp_noise)

        Since exp_mp_noise vs exp_mp_std looks like a line, I define the simulation noise to be also linear fit in such a way that the ratio of exp_mp_nosie to exp_mp_STD is the same as sim_mp_noise to sim_mp_STD

        output:
        -------
            noise_model:        polyfit 1d, property gets added to the cell object 
            
        output:
        -------
            sim_noise_fit:      A linear fit to sim_noise_sd vs sim_mp_sd
                                Both sim_mp_sd and sim_noise_sd are in the same units as the simulation
                                Given a set of linear prediction, compute the STD and the STD of the noise to use is sim_noise_fit( linear_prediction.std() )
        '''
        #_pdb.set_trace()

        # load parameters from text file
        #global bipolar_cell_file
        df = _pd.read_csv(bipolar_cell_file, sep=' ').sort('exp_contrast').reset_index(drop=True)#, index_col='exp_contrast')
        
        # Simulate the experimental data
        for i, contrast in enumerate(df['exp_contrast']):
            # compute the sd of the simulated membrane potential when using a gaussian distribution signal of the same contrast as the one used in the real experiment
            df.set_value(i, 'sim_mp_sd', self.sim_central_pathway('gaussian', contrast).std())

            # now scale the noise such that the ratio between noise/mp_sd is the same in the experiment and in the simulation
            df.set_value(i, 'sim_mp_noise', df.get_value(i, 'sim_mp_sd')*df.get_value(i, 'exp_mp_noise')/df.get_value(i, 'exp_mp_sd'))

        # fit a line between all the values of sim_mp_noise and sim_mp_sd
        sim_noise_fit = _np.poly1d(_np.polyfit(df['sim_mp_sd'], df['sim_mp_noise'],1))

        # the connection between the experiment and the simulation are the gaussian contrasts. For each gaussian contrast, express the noise as a fraction 
        if plot_flag:
            self.plot_noise_model()

        return sim_noise_fit
        
    def add_mp_noise(self, mp, integration_time):
        '''
        Add noise to the membrane potential 'mp'. 
        
        Algorithm depends heavily on the integrationg_time value:

        integration_time>0:         Noise model is computed from Yusuf's recordings in get_mp_noise_model (which adds property 'noise_model' to cell object)
                                    To compute the sd of the membrane potential a sliding window of length 'integration_time' is used
        integration_time == 0       The SD at each point is taken across cells without combining different times.
                                    mp has to be 2D and mp.std(axis=0) is used

        integration_time == -1      Just compute the SD of mp. A single noise value is used for all mp

        if mp is 2D, the 1st points of every row are unreliable. If mp is 1D, the 1st points are unreliable.
        By making mp 1D, computing the running SD and then converting it back to its original shape
        '''
        #_pdb.set_trace()

        # if cell has no attribute noise_model, compute it. This will take a little time since it has to open a file and makes a graph but is only done once
        #if not hasattr(self, 'noise_model'):
        #    self.noise_model = reproduce_Yusuf()

        # Depending on the value of integration_time, compute the SD of mp that is needed in order to generate the random noise
        # Then compute the noise. Noise has to be of the same shape as mp to be added at the end.
        if integration_time > 0:
            sd = _np.zeros_like(mp.flatten())
            
            integration_points = int(integration_time/sim_delta_t)
            for i in range(integration_points, len(sd)):
                sd[i] = mp.flatten()[i-integration_points:i].std()
            
            sd = sd.reshape(mp.shape)

            # in this case sd is already teh same size as mp. Generate an array of noise the same size as mp with a SD of 1 and multiply each value by its corresponding SD
            noise = _np.multiply(_np.random.standard_normal(mp.shape), sd)

        elif integration_time == 0:
            sd = mp.std(axis=0)

            # in this case, sd is 1 row with as many columns as mp
            noise = _np.random.standard_normal(mp.shape) * sd

        elif integration_time ==-1:
            sd = mp.std()

            # in this case, sd is a single value
            noise = _np.random.standard_normal(mp.shape) * sd

        return mp +.001* noise

    def get_noise(self, shape, std, corr_time, save_flag=0):
        '''
        this is not well tested. I'm trying it out

        Generate a pink noise sequence of the given 'shape'.
        Noise is generated as 1D with zero mean and stadnard deviation given by 'std' and at the end it is reshaped to the given shape.
        I'm also gnerating the pink noise such that it has an auto correlation of roughly corr_time (assuming as in the simulation a step given by 'sim_delta_t')
        '''
        #_pdb.set_trace()
        try:
            sacc_path = 'Sacc={0}, RWS=.0{1}/'.format(saccade_size, int(rw_step*100))
            noise = _np.fromfile(sacc_path + 'linear_prediction_noise').reshape(shape)
            print('Noise loaded from "linear_prediction_noise" file')
        except:
            if corr_time == sim_delta_t:
                noise = _np.random.randn(*shape)                        # here 'np' is numpy, not a typo
            else:
                noise = _pn.pink(shape, depth=corr_time/sim_delta_t)    # here 'pn' is pinknoise, not a typo

            #_pdb.set_trace()
            noise -= noise.mean()
            noise *= std/noise.std()
            if save_flag:
                N = 14
                print('*'*72)
                print('*'+' '*N+'Saving linear_prediction_noise'+' '*N+'*')
                print('*'*72)

                noise.tofile(sacc_path + 'linear_prediction_noise')

        return noise

    def sim_central_pathway(self, stim_type, contrast, length=500, mean=127):
        '''
        Simulate responses to either 'pink' or 'gaussian' experiment.
        In this case there is no need to filter spatially since stimulus is all the same in space.
        

        input:
        ------
            stim_type:      'pink' or 'gaussian'

            length:         in seconds
        '''
        # Since combolution will remove some points from stim, request a stim of length such that after convolution and rmoving the extra points the response will be of the desired length
        #_pdb.set_trace()
        ker_length = len(self.center.kernel)*sim_delta_t
        stim = fake_noise(stim_type, contrast, length=length+2*ker_length, mean=mean)
        
        resp = self.center.temporal_filter(stim) + self.surround.temporal_filter(stim)

        samples = int(length/sim_delta_t)
        return resp[:samples]

    def add_peripheral_pathway(self, central_mp, peripheral_weight, psth_pnts, amp_noise_SD=None):
        '''
        compute and add the contribution of gating to the membrane potential

        Peripheral pathway will be the same size as last dimension in central_mp

        input:
        ------
            amp_noise_SD:       SD of peripheral amplitud noise
                                Instead of adding always peripheral_weight*self.periphery.kernel, the amplitued is modulated by noise as a gaussian process around peripheral_weight with standard deviation given by peripheral_weight * amp_noise_SD
                                The amplitud of peripheral imput is: randn()*peripheral_weight*amp_noise_SD + peripheral_weight
        '''

        #_pdb.set_trace()

        last_dim_size = central_mp.shape[-1]
        
        # limit peripheral kernel to be psth_pnts and change it to be (1, psth_pnts) (for matrix multiplication later on)
        peri_kernel = self.periphery.kernel[:psth_pnts].reshape(1, -1)

        # N is the number of times I have to concatenate peri_kernel to get as many points as last_dim_size
        N = _np.ceil(last_dim_size/psth_pnts)

        # make an array with the peripheral random amplitudes to use.
        trials = _np.ceil(central_mp.size/psth_pnts)
        amp = _np.random.randn(trials, 1)*peripheral_weight*amp_noise_SD + peripheral_weight

        # Matrix multiply each amplitud by the peripheral kernel.
        gating_mp = amp*peri_kernel

        # if gating_mp has more points than central_mp, remove excess
        gating_mp = gating_mp[:central_mp.size]
        gating_mp = gating_mp.reshape(central_mp.shape)

        return central_mp + gating_mp


    def get_gating_letters(self, lp):
        '''
        Pass linear prediction 'lp', through the corresponding nonlinearity to get gating latters. Corresponding nl is a combination of nl_basal plus a shift, where the shift is given by peripheral.kernel

        lp:     2D ndarray where lp[i,j] represents cell i, point in time j (and j = 0 corresponds to sim_start_t and j=-1 to sim_end_t)

        At this point, lp time is in sim_delta_t and NOT in letter_length
        '''
        #_pdb.set_trace()

        # grab peripheral contribution
        kernel = self.periphery.kernel*nl_gating_amplitud
        """
        _plt.figure()
        tax = _np.arange(0, len(kernel)*.005, .005)
        _plt.plot(tax, kernel)
        """
        # preallocate gating letters
        gating_letters = _np.zeros_like(lp)

        # loop through the time axis of lp (axis=1), compute the nl according to the threshold and pass lp through it
        for p in range(lp.shape[1]):
            # convert p from lp to a shift.
            # first convert p to time in lp and constrain it to be in the kernel range, then convert it to a shift
            t = p*sim_delta_t + sim_start_t
            if t < 0 or t >.45:         # not important to distinguish between less than 0 or more than .45 since both have the same shift (0)
                t = 0

            kernel_pnt = t/sim_delta_t
            shift = kernel[kernel_pnt]

            nl = self.nl_basal.copy()
            nl.thresh -= shift              # periphery kernel is designed to be excitatory during gating. Since I'm modeling it as a threshold shift, the threshold has to shift in the opposite direction (decreasing to get excitation)
            gating_letters[:,p] = nl.torate(lp[:,p])

        return gating_letters

    def get_ca_concentration(self):
        self.nl.torate


    def simulate_UFlicker(self, mean, contrast, trials, peri_factor = 1, TWN=10, nl_pnts=20, plot_flag=1):
        '''
        fake gaussian stimuli of given contrast and pass it through the model.

        At the end, compute one PSTH and dividing lp and vesicles into tw of length 50ms each and compute kernel and nl per tw

        input:
            mean/contrast:      parameters to define Gaussian stimulation

            trials:             each trial lasts len(self.periphery.kernel)*sim_delta_t seconds

            peri_factor:        float design to turn on/off periphery contribution

        output:
            PSTH:

            kernels:            list of kernels, one per TW     

            nls:                list of nls, one per TW

        '''
        #_pdb.set_trace()

        # I want to simulate the central pathway for N trials each lasting the same as periphery_kernel
        kernel_pnts = len(self.periphery.kernel)
        length = trials * kernel_pnts * sim_delta_t

        # simulate stimulus, # has to be longer than length because convolution later on will make it shorter
        ker_length = kernel_pnts * sim_delta_t
        stim = fake_noise('gaussian', contrast, length=length+ker_length+1, mean=mean)
        
        # simulate linear response
        samples = int(length/sim_delta_t)
        lp_center = self.center.temporal_filter(stim) + self.surround.temporal_filter(stim)
        stim = stim[kernel_pnts-1:]         # now stim and lp_center are aligned

        # remove extra points and reshape to be (trials, kernel_pnts)
        lp_center = lp_center[:samples].reshape(-1, kernel_pnts)
        stim = stim[:samples].reshape(-1, kernel_pnts)

        # add peipheral pathway. In next line, kernel is 1D and lp is 2D but summation works fine
        lp = lp_center + self.periphery.weight * self.periphery.kernel * peri_factor

        # generate [Ca] from lp, using fixed nl (basal, all periphery effects are taken into account in the lp)
        letters = self.nl_basal.torate(lp)

        # generate vesicle release by adapting [Ca]
        vesicles = self.adaptation.adapt(letters)

        #""" Good for debugging
        vesicles1D = vesicles.reshape(-1)
        stim1D = stim.reshape(-1)

        # generate PSTH from vesicles. 
        PSTH = vesicles.mean(axis=0)

        """ Good for debuging. Computes one kernel and one nl from computed vesicles
        # compute kernel from the vesicles and the stimulus
        #_pdb.set_trace()
        kernel = compute_kernel(stim1D, vesicles1D, kernel_pnts, 0)
        ker_object = filter_block(1, kernel, 1)
        recovered_lp = ker_object.temporal_filter(stim1D)
        vesicles1D = vesicles1D[kernel_pnts-1:]
        nl = compute_nl(recovered_lp, vesicles1D,100)

        #_plt.close('UFlicker')
        fig, ax = _plt.subplots(num='UFlicker', nrows=2)
        ax[0].plot(kernel)
        ax[1].plot(nl[0], nl[1])

        ax[1].set_ylim(-.5, ax[1].get_ylim()[1])
        """
        # now compute 2 kernels and nls. One with the 1st half of the data and other with the 2nd half
        kernels = []
        nls = []
        #_pdb.set_trace()
        for i in range(TWN):
            # define time window
            tw_startP = i*kernel_pnts/TWN
            tw_endP = (i+1)*kernel_pnts/TWN

            # Define an array like vesicles but with zeros outside the TW in question
            tw_vesicles = _np.zeros_like(vesicles)
            tw_vesicles[:, tw_startP:tw_endP] = vesicles[:, tw_startP:tw_endP]
            tw_vesicles = tw_vesicles.flatten()

            # Define the Filtering block
            kernel = compute_kernel(stim1D, tw_vesicles, kernel_pnts, 0)
            kernels.append(kernel)
            ker_object = filter_block(1, kernel, 1)

            # Convolve stim with kernel and remove 1st points from tw_vesicles to align it with the recovered_lp
            recovered_lp = ker_object.temporal_filter(stim1D)
            tw_vesicles = tw_vesicles[kernel_pnts-1:]

            # Downsample a few points recovered_lp and tw_vesicles before computing nl. I think adaptation effects prefent the cell from responding at the highest lps and nl comes down.
            recovered_lp = average(recovered_lp, 10, 4)
            tw_vesicles = average(tw_vesicles, 10, 4)

            # compute nl
            nls.append(compute_nl(recovered_lp, tw_vesicles, nl_pnts))

        if plot_flag:
            _plt.close('UFlicker2')
            fig, ax = _plt.subplots(num='UFlicker2', nrows=2, ncols=TWN, sharey='row')
            for col in range(TWN):
                ax[0][col].plot(kernels[col])
                ax[1][col].plot(nls[col][0], nls[col][1])
                if col > 0:
                    ax[1][col].twiny

        return PSTH, kernels, nls


    def plot_gaussian_simulation_and_nls(self, g, contrast, times, nls):
        '''
        This might not need to be a cell method but too lazzy to change it.

        Plots g values at the given times (might be empty)

        Plots filtered gaussian stimulus by cell 'self'. contrast can be one float or a list, can be empty list as well

        Plots also all nonlinear objects in nls. Can be an empty list.
        '''
        #_pdb.set_trace()
        if not _np.iterable(contrast):
            contrast = [contrast]

        if not _np.iterable(times):
            times = [times]

        if not _np.iterable(nls):
            nls = [nls]

        _plt.close('gaussian_simulation_and_nls')
        fig, ax1, = _plt.subplots(num='gaussian_simulation_and_nls')
        
        bins = 200
        # simulation data
        colors = ((0,.75,0), (.75,.75,0))
        for i, t in enumerate(times):
            point = time_to_point(t,0)
            data_to_hist = g[:,point]
        
            label = r'$t ={0: G}ms$'.format(int(1000*t))
            #hist, bins, patches = ax1.hist(data_to_hist, color=(.6, .6, .6), bins=bins, normed=True, histtype='stepfilled', alpha=.995, label=label)
            hist, bins, patches = ax1.hist(data_to_hist, color = colors[i], bins=bins, normed=True, histtype='stepfilled', alpha=.5, label=label)
        
        # Gaussian, like in UFLicker
        for C in contrast:
            gaussian_g = self.sim_central_pathway('gaussian', C)
            ax1.hist(gaussian_g, color=(0,1,1), bins = bins, histtype='stepfilled', normed=True, alpha=.5, label=r'${0}\%\, Contrast$'.format(C))

        if len(nls):
            ax2 = ax1.twinx()
        
        for i, nl in enumerate(nls):
            if len(nls) in [2,3] and i==0:
                label = r'$basal$'
                color = 'r'
            elif len(nls) in [2,3] and i==1:
                label = r'$gating$'
                color = 'b'
            elif len(nls) in [2,3] and i==2:
                label = r'$inhibiting$'
                color = ':b'
            else:
                label = '_nolegend_'
                color = 'g'

            ax2.plot(bins, nl.torate(bins), color, lw=2, label=label)

        ax1.set_axis_off()
        ax2.set_axis_off()
        ax1.set_yticks([])
        ax2.set_yticks([])
        ax1.set_xticks([])
        
        ax1.set_xlim(-300,300)
        #ax2.set_ylim( (0,ax2.get_ylim()[1]*.65) )
        ax2.set_ylim((0, 500))
        ax1.legend(fontsize=10, bbox_to_anchor=(1.1,1), handlelength=1, frameon=False)
        if len(nls):
            ax2.legend(loc='upper left', fontsize=10, handlelength=1, frameon=False)


        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        fig.subplots_adjust(left=0, bottom=0, right=.9, top=1)
        fig.set_size_inches(2.5,2)

        fig.savefig('Figures/gaussian_simulation_and_nls.pdf', transparent=True)

        return fig


    def plot_noise_model(self):
        '''
        Plot the ratio between exp_mp_noise and exp_mp_sd as a function of stimulus contrast
        '''
        _plt.close('noise_model')
        fig, ax = _plt.subplots(num='noise_model')

        df = _pd.read_csv(bipolar_cell_file, sep=' ').sort('exp_contrast').reset_index(drop=True)#, index_col='exp_contrast')
        print(df.columns)
        #ratio = df['exp_mp_noise']/df['exp_mp_sd']
        ratio = df['exp_mp_sd']/df['exp_mp_noise']
        ax.plot(df['exp_contrast'], ratio, 'ok')
        fit = _np.poly1d(_np.polyfit(df['exp_contrast'], ratio,1))
        ax.plot([0,.4], [fit(0), fit(.4)], 'k')

        # formating figure
        #ax.set_axis_off()
        ax.set_xlabel(r'$Contrast$', fontsize=10, labelpad=0)
        ax.set_ylabel(r'$SNR$',fontsize=10, labelpad=-1)
        xticks = _np.arange(0, .5, .2)
        ax.set_xticks(xticks)
        ax.set_xticklabels((0, 20, 40), fontsize=10)    # actual x ticks are 0, .2 and .4 but I'm displaying it as %
        yticks = range(0, 10, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        #ax.xaxis.set_ticks_position('bottom')
        #ax.yaxis.set_ticks_position('left')
        ax.set_ylim(.0, ax.get_ylim()[1]*1.1)
        fig.subplots_adjust(bottom=.35, left=.2, right=1, top=1)
        fig.set_size_inches(1.5,1)
        ax.tick_params(axis='both', length=3,right='off', top='off')
        fig.savefig('Figures/noise_model.pdf', transparent=True, pad_inches=0)
        return fig


    def plot_noise(self, g, fig_g):
        ''' 
        plot a few examples of the noise added to the mp and the STD that generated them.

        fig_g is the fig handle that comes out of plot_g. I'm using it here to set the y axis identical to that of plot_g
        '''

        #_pdb.set_trace()

        _plt.close('noise')
        fig, ax = _plt.subplots(num='noise')

        tracesN = 30
        std = self.noise_model(g.std(axis=0))
        noise = self.get_noise(g.shape, std, noise_corr_time)
        tax = _get_simulation_TAX()

        for i in range(tracesN):
            ax.plot(tax, noise[i,:], color='#BBBBBB', alpha=.2)

        ax.plot(tax, std, 'k', lw=2)


        ax.set_xlim(-.2, .8)
        ax.set_xlabel(r'$Time\, (s)$')
        ax.set_ylabel(r'$V_m\, Noise$', labelpad=-10)

        xticks = _np.arange(-.2, .8, .4)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=10)

        # plot y axis to be 1/10 as in fig_g
        fig_g_axes = fig_g.get_axes()[0]
        yticks = fig_g_axes.get_yticks()
        yticks = (yticks[0]/10, yticks[1]/10)
        ylim = (1.1*yticks[0], 1.1*yticks[1])
        ax.set_yticks(yticks)
        ax.set_yticklabels([-0.1, 0.1], fontsize=10)
        ax.set_ylim((ylim[0], ylim[1]))

        # add doted line at time = 0
        ax.plot([0,0], ax.get_ylim(), ':k', label='_nolegend_')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(length=3, right='off', top='off')

        fig.subplots_adjust(left=.25, bottom=.35, top=.95, right=1)
        #fig.subplots_adjust(bottom=.35, left=.2, right=1, top=1)
        fig.set_size_inches(2, 1.5)
        fig.savefig('Figures/nosie.pdf', transparent=True)
        
        return fig

    def plot_noise_correlation(self, g):
        '''
        plot autocorrelation of noise
        '''

        _plt.close('noise_correlation')
        fig, ax = _plt.subplots(num='noise_correlation')

        N = 10000
        noise = self.get_noise((N,1), 1, noise_corr_time)

        corr = _np.correlate(noise.flatten(), noise.flatten(), mode='full')

        tax = _np.linspace(-(N-1)*sim_delta_t, (N-1)*sim_delta_t, 2*N-1)

        ax.plot(tax[9900:10100], corr[9900:10100], 'ok')
        ax.set_xlim(-2*noise_corr_time, 2*noise_corr_time)
        ax.set_ylim(-500, 10500)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.savefig('Figures/noise_correlation.pdf', transparent=True)
        
        return fig

    def _test_filtering(self, mean):
        '''
        Test that sim_central_pathway (avoiding spatial filtering) gives the same result had I used the spatial filtering.
        
        generate a 'mean' intensity image and filter it with both center and surround, then generate stim by concatenating those values and pass them to temporal filtering

        On the other hand, send to sim_central_pathway a stim with 0 contrast and the same mean
        '''
        sim1 = self.sim_central_pathway('gaussian', 0, mean=mean)

        image = _loadImage(0)
        image_1 = _np.ones_like(image)*mean
        self.center.filter_image(image_1)
        self.surround.filter_image(image_1)

        sim2 = self.center.temporal_filter([self.center.filtered_image[500,500]]*1000) + self.surround.temporal_filter([self.surround.filtered_image[500,500]]*1000)

        return sim1, sim2
    
    
    def simulate_PSTH(self, peri_weight, nl_thresh, avg_fr, stim_type, contrast, mean, trials, psth_pnts, central_LP=None):
        '''
        simulate the PSTH.
        
        1. Compute central linear prediction for the given type of stim. Signal will have stim_length samples.
        2. Combine central linear prediction with a scaled version of peripheral signal (scale factor is peri_weight).
        3. Threshold noisy linear prediction with nl_thresh.
        4. average all trials.
        
        input:
        ------

            peri_weight:    peripheral pathway gets scaled by this value before being combined with central linear prediction.

            nl_thresh:      will be used to threshold data instead of using the cell's nl_block

            stim_type:      'pink' or 'gaussian'

            trials:         number of trials to simulate, each lasting sim_delta_t*psth_nts

            psth_pnts:      number of points in the simulated PSTH. The simulation is done with sim_delta_t seconds in between points

        '''
        #_pdb.set_trace()
        # fake the central pathway, unless given
        if central_LP is None:
            central_LP = self.sim_central_pathway(stim_type, contrast, mean=mean, length=trials * sim_delta_t * psth_pnts)
        
            # reshape central_LP such that there are many trials each lasting psth_pnts
            central_LP = central_LP.reshape(-1, psth_pnts)

        # add scaled version of peripheral input to central_LP
        lp = self.add_peripheral_pathway(central_LP, peri_weight, psth_pnts, amp_noise_SD=.10)

        # add noise to lp
        noise = self.add_mp_noise(lp, 0)
        #noisy_lp = self.add_mp_noise(lp, -1)
        noisy_lp = noise + lp

        # threshold lp
        noisy_lp = noisy_lp - nl_thresh
        below_threshold_values = noisy_lp < 0
        noisy_lp[below_threshold_values] = 0

        # compute psth and divide by a average activity
        #memory = 5
        psth = noisy_lp.mean(axis=0)
        #smoothing_ker = _np.ones(memory)/memory
        #smoothed = _np.convolve(psth, smoothing_ker)
        #psth = _np.divide(psth, smoothed[:-memory+1])
        psth_mean = psth.mean()

        if psth_mean != 0:
            psth *= avg_fr/psth.mean()

        return psth, central_LP#, lp, noise, noisy_lp

    def plot_exp_and_simulated_PSTH(self, psth, peri_weight, nl_thresh, avg_fr, trials=100):
        '''
        plot the given psth along with the simulated version using peri_weight and nl_thresh
        '''

        #_pdb.set_trace()

        # get the simulated psth

        psth_sim, _ = self.simulate_PSTH(peri_weight, nl_thresh, avg_fr, 'pink', .1, 127, trials, len(psth))
        tax = _np.arange(0, len(psth)*sim_delta_t, sim_delta_t)

        _plt.close('exp_and_simulated_PSTH')
        fig, ax = _plt.subplots(num='exp_and_simulated_PSTH')

        ax.plot(tax, psth, lw=2, label=r'$data$')
        ax.plot(tax, psth_sim, lw=2, label=r'$simulation$')

        xticks = _np.arange(0, .5, .25)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, size=10)
        yticks = range(0, 8, 2)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, size=10)

        ax.set_xlabel(r'$Time\, (s)$', size=12)
        ax.set_ylabel(r'$Rate\, (Hz)$', size=12, labelpad=0)

        ax.tick_params(length=3, right='off', top='off')

        ax.legend(loc='upper right', bbox_to_anchor = (1, 1), fontsize=10, handlelength=1, frameon=False)
        fig.subplots_adjust(bottom = .30, left=.2,top=1,right=.95)
        

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.set_size_inches(2.5,2)
        fig.savefig('Figures/exp_and_simulated_PSTH.pdf', transparent=True)

        return fig

    def _callback(self, event):
        import sys
        print('clicked: ', event)
        sys.stdout.flush()

    def _fit_PSTH(self, exp_psth, stim_type, contrast, mean, trials, psth_pnts, peri_range, thresh_range, plot_flag=0):
        '''
        fit peri_weight and nl_thresh to get a good estimate between simulate_PSTH and the given experimental PSTH.

        input:
        ------
            exp_psth:       if mixing different mean/contrasts exp_psth is already a concatenation of all the psths each lasting psth_pnts

            styp_type:      'gaussian' or 'pink'

            contrast:       iterable of contrasts

            mean:           iterable of means

            trials:         how many trials to simulate, total simulation will last trials * sim_delta_t * psth_pnts

            ptsh_pnts:      how many points does each psth last. If using more than one condition exp_psth has all those psths concatenated togehter. In that case psth_pnts is the length of each psth. len(exp_psth) = psth_pnts * len(contrast)

            peri_ragne:     range object or iterable

            thresh_range:   range object or iterable
        '''

        # check that all input is appropriate
        if not _np.iterable(contrast):
            contrast = [contrast]

        if not _np.iterable(mean):
            mean = [mean]

        if len(mean) != len(contrast):
            raise ValueError('cell._fit_PSTH: mean and contrast should be the same length')

        if len(exp_psth)!= len(mean)*psth_pnts:
            raise ValueError('cell._fit_PSTH: len(exp_psth) should be equal to len(mean)*psth_pnts')

        #_pdb.set_trace()

        # redefine peripheral kernel such that gatig window alligns with experimental gating
        gating_start_t, gating_end_t = self.redefine_gating_window(exp_psth)
        peri_kernel = generate_peripheral_kernel(gating_start_t, gating_end_t, psth_pnts)
        self.periphery = filter_block(periphery_size, peri_kernel, periphery_weight)
        
        LP = ()
        psths = ()
        # make a first call to get central_LP
        for i in range(len(contrast)):
            _, central_LP = self.simulate_PSTH(0, 0, 1, stim_type, contrast[i], mean[i], trials, psth_pnts)
            LP = LP+(central_LP,)

        errors = []
        error = _np.infty
        for i in range(len(contrast)):
            c = contrast[i]
            m = mean[i]
                
            # I'm requesting that the average firing rate of simulated psth be the same as the experimental psth.
            avg_fr = exp_psth[i*psth_pnts:(i+1)*psth_pnts].mean()

            for peri in peri_range:
                for thresh in thresh_range:
                    psth, _ = self.simulate_PSTH(peri, thresh, avg_fr, stim_type, c, m, trials, psth_pnts, central_LP = LP[i])
                    psths = psths + (psth,)

                    # concatenate the different psths together
                    psth = _np.concatenate(psths, axis=0)


                    # exp_psth:     if mixing different mean/contrasts exp_psth is already a concatenation of all the psths each lasting psth_pnts
                    new_error = (psth-exp_psth).std()
                    #new_error = new_error.mean()
                    errors.append(new_error)
                    if new_error < error:
                        error = new_error
                        best_params = (peri, thresh, avg_fr, error)
                    
                    # get ready for another loop
                    psths = ()

        if plot_flag:
            self.plot_exp_and_simulated_PSTH(exp_psth, best_params[0], best_params[1], best_params[2])
            #_plt.close('fit-errors')
            #fig, ax = _plt.subplots(num='fit-errors')
            #ax.plot(errors)

        return best_params#, errors


    def redefine_gating_window(self, exp_psth, plot_flag=0, save_flag=0):
        '''
        redefine peripheral kernel such that gatig window alligns with experimental gating
        '''
        #_pdb.set_trace()
        
        # define a threshold to identify gating start/stop. Threshold is midpoint between starting firing rate and the maximum before 0.2 seconds
        start_fr = exp_psth[:5].mean()
        threshold = (start_fr + exp_psth[5:.2/sim_delta_t].max())/2

        p_of_max = _np.where(exp_psth>threshold)[0][0]

        p_of_min = _np.where(exp_psth[p_of_max:] < threshold)[0][0]

        gating_start_t = p_of_max * sim_delta_t#- .01
        gating_end_t = (p_of_max + p_of_min) * sim_delta_t# - .01

        return gating_start_t, gating_end_t
        """
        peri_kernel = generate_peripheral_kernel(len(exp_psth), save_flag=save_flag)
        
        self.periphery = filter_block(periphery_size, periphery_kernel_file, periphery_weight)

        if plot_flag:
            tax = _np.arange(0, len(exp_psth)*sim_delta_t, sim_delta_t)

            _plt.close('new_peri_kernel')
            fig, ax = _plt.subplots(num='new_peri_kernel')

            ax.plot(tax, exp_psth)
            ax.plot(tax, self.periphery.kernel*exp_psth.max())
        """


