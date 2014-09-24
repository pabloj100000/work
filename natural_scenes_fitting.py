'''
I started this file as an attempt to fit PSTHs by the model, the original example showing how leastsq fit works that I was following was modified from http://wiki.scipy.org/Cookbook/FittingData

'''
import pdb as _pdb
import numpy as _np
import matplotlib.pyplot as _plt
from scipy import optimize
import naturalscenes as _ns
import pandas as _pd

def _leastsq_test():
    '''
    some modifications to the example extracted from http://wiki.scipy.org/Cookbook/FittingData
    '''
    num_points = 150
    x = _np.linspace(5., 8., num_points)

    y = 11.86*_np.cos(2*_np.pi/0.81*x-1.32) + 0.64*x+4*((0.5-_np.random.rand(num_points))*_np.exp(2*_np.random.rand(num_points)**2))
    
    #Fitting the data
    # We now have two sets of data: x and Ty, the time series, and tX and tY, sinusoidal data with noise. We are interested in finding the frequency of the sine wave.
    # Fit the first set
    def fitfunc(p, x):
        return p[0]*_np.cos(2*_np.pi/p[1]*x+p[2]) + p[3]*x # Target function

    def errfunc(p, x, y):
        return fitfunc(p, x) - y # Distance to the target function
    
    p0 = [-15., 0.8, 0., -1.] # Initial guess for the parameters
    p1, success = optimize.leastsq(errfunc, p0[:], args=(x, y))
    print(p1)

def get_all_arguments(cell_id):
    '''
    every time I call fitfunc I need to pass many parameters, some of which need to be computed and take time like noise_model (from reproduce_Yusuf). Here I have packed all those functions such that this function is called once and then fitfunc can be called many times as follows:
    
    args = get_all_arguments(cell_id)
    fitfunc(periphery_weight, nl_thresh, *args, [plot_flag])
    
    inptus:
    -------
        cell_id (int):      an integer, it should help identify UFlicker PSTHs.
                            psths are stored in: UFlicker PSTHs/UFlicker_PSTH_c'cell_id'_'contrast'c.txt

    output:
    -------
        if PSTHs were found...
            x:                  an x axis that would be obtained out of concatenating all the x axis of each individual PSTH
            
            psth_pnts (int):    number of points in each psth, they have to have the same number of points
            
            contrasts:          iterable of contrasts loaded
            
            means:              iterable of means used, for the time being they are all the same
            
            nose_model          output of reproduce_Yusuf()
            
            psths               actual concatenation of all psths

        if PSTHs were not found...
            None

    '''
    #_pdb.set_trace()

    loaded_PSTH = _ns.load_UFlicker_PSTH(cell_id)
    if loaded_PSTH is None:
        return None

    x, psths, psth_pnts = loaded_PSTH

    contrasts = [3, 6, 12, 24]
    means = [127, 127, 127, 127]

    if len(x)/psth_pnts==5:
        contrasts.append(100)
        means.append(127)

    noise_model, _ = _ns.reproduce_Yusuf(plot_flag=0)

    return x, psth_pnts, contrasts, means, noise_model, psths
    
def errfunc(periphery_weight, nl_thresh, x, psth_pnts, contrasts, means, noise_model, psths):
    '''
    Just compute the average abs error in approximating the psth by the model
    '''
    return abs(fitfunc(periphery_weight, nl_thresh, x, psth_pnts, contrasts, means, noise_model, psths) - psths).mean()

#Fitting the data
# We now have two sets of data: x and Ty, the time series, and tX and tY, sinusoidal data with noise. We are interested in finding the frequency of the sine wave.
# Fit the first set
def fitfunc(periphery_weight, nl_thresh, x, psth_pnts, contrasts, means, noise_model, psths, plot_flag=0):
    '''
    Compute the model corresponding to periphery_weight and nl_thresh for all contrasts and means and taking noise_model into account

    output:
        returns the modeled psth or Ca concentration

    '''
    #periphery_weight, nl_thresh, scale_factor = p

    # convert x to mean and contrast

    # Define all pathways
    center_path = _ns.filter_block(_ns.center_size, _ns.center_kernel_file, _ns.center_weight)
    surround_path = _ns.filter_block(_ns.surround_size, _ns.surround_kernel_file, _ns.surround_weight)
    periphery_path = _ns.filter_block(_ns.periphery_size, _ns.periphery_kernel_file, _ns.periphery_weight)
    periphery_path.kernel = periphery_path.kernel[:psth_pnts]

    stim_length = 1000*psth_pnts+1
    
    mps = []
    for m, c in zip(means,contrasts):
        # simulate that mean and contrast
        stim = _ns.fake_gaussian_noise(c, mean=m, samples = stim_length)
        
        # pass gaussian stimuli through center and surround pathways (not peripheral one)
        mp = center_path.temporal_filter(stim) + surround_path.temporal_filter(stim)

        # before proceeding make sure each mp has a number of points that is an integer number of psths
        N = int(len(mp)/psth_pnts)*psth_pnts
        mp = mp[:N]

        # reshape mp such that each row represents one peripheral kernel (one period). Then add it to list mps
        mps.append(mp.reshape(-1, psth_pnts))


    # concatenate all mps such that each row represensts one trial of all contrasts and means
    mp = _np.concatenate(mps, axis=1)

    # add the impulse response coming from the background to generate a gated version of mps
    # concatenate one impulse response after another spanning the whole simulation. I'm actually making periphery_mp bigger than mps (that's the +1) but then I'm trashing all extra points
    #periphery_mps = [_np.concatenate((periphery_path.kernel,)*int(mp.shape[0]/periphery_path.kernel.shape[0]+1))[:mp[0].shape[0]] for mp in mps]
    periphery_mp = _np.concatenate((periphery_path.kernel,)*len(mps))

    # compute noise by passing total_mp through noise_model
    gated_mp = mp + periphery_weight*periphery_mp

    # compute noise model according to intracellular recordings and add it to simulated mp. At each time point, compute the SD of gated_mp and use it to scale the random noise
    gated_noisy_mp = gated_mp + _np.random.standard_normal(gated_mp.shape) * noise_model(gated_mp.std(axis=0))

    # pass noisy mp through nonlinearity describing [Ca] 
    gated_ca_conc = gated_noisy_mp - nl_thresh
    below_thresh_indices = gated_ca_conc < 0
    gated_ca_conc[below_thresh_indices] = 0
    gated_ca_conc = gated_ca_conc.mean(axis=0)
    
    #_pdb.set_trace()

    # adapt [Ca] dividing by its mean
    adapted_ca_conc = gated_ca_conc[:]
    for i in range(len(means)):
        # since UFlicker and Stable object come from different cells I use different scaling factors
        """
        if i <len(means)/2:
            scale_factor = scale_factor1
        else:
            scale_factor = scale_factor2

        """
        adapted_ca_conc[i*psth_pnts:(i+1)*psth_pnts] /= gated_ca_conc[i*psth_pnts:(i+1)*psth_pnts].mean()

    # constrain adapted_ca_conc to have the same mean as psths
    adapted_ca_conc *= psths.mean()/adapted_ca_conc.mean()

    if plot_flag:
        _plt.close('fit_test')
        fig, ax = _plt.subplots(num='fit_test')
        ax.plot(x, psths)
        ax.plot(x, adapted_ca_conc)
        #fig, ax = _plt.subplots(nrows=2, num='fit_test')
        #half = len(x)/2
        #ax[0].plot(x[:half], psths[:half])
        #ax[0].plot(x[:half], adapted_ca_conc[:half])
        #ax[1].plot(x[half:], psths[half:])
        #ax[1].plot(x[half:], adapted_ca_conc[half:])

    return adapted_ca_conc

def loop_params(cell_id_or_args, peri_weight_range, thresh_range, flag=None, depth=0):
    '''
    Loop over some params and report the values that give the best [Ca] estimate
    
    So far those are periphery_weight = 47, nl_thresh=32

    input:
    -------
        cell_id_or_args:    either an int representing a cell or a list of arguments as returned by test4()

        flag:           None: optimizing both parameters
                        1:      optimizing only peri_weight
                        2:      optimizing only thresh

        depth:          should not be used when calling this function. Measures the depth in recurssion

    output:
    -------
        best_params:    tuple with (peri_weight, thresh, error)

        args:           arguments other than peri_weight and thresh needed in calling funcfit
    '''
    params = []
    error = []
    max_depth = 5

    #_pdb.set_trace()
    if isinstance(cell_id_or_args, int):
        args = get_all_arguments(cell_id_or_args)
    else:
        args = cell_id_or_args

    #print('args = ', args)
    if args is None:
        return None

    for peri_weight in peri_weight_range:
        for thresh in thresh_range:
            params.append((peri_weight, thresh))
            error.append(errfunc(peri_weight, thresh, *args))

    best_params = params[error.index(min(error))] + (min(error),)

    if depth>=max_depth:
        return best_params, args

    if flag != 2:
        if best_params[0] == peri_weight_range[0]:
            print('peri_weight too low', best_params[0], peri_weight_range)
            delta = peri_weight_range[1]-peri_weight_range[0]
            new_min = peri_weight_range[0]-5*delta
            new_max = peri_weight_range[0]+1*delta
            new_range = range(new_min, new_max, delta)
            #_pdb.set_trace()
            best_params, _ = loop_params(cell_id_or_args, new_range, thresh_range, flag = flag, depth=depth+1)
        elif best_params[0] == peri_weight_range[-1]:
            print('peri_weight too hi', best_params[0], peri_weight_range)
            delta = peri_weight_range[1]-peri_weight_range[0]
            new_min = peri_weight_range[-1]-1*delta
            new_max = peri_weight_range[-1]+5*delta
            new_range = range(new_min, new_max, delta)
            #_pdb.set_trace()
            best_params, _ = loop_params(cell_id_or_args, new_range, thresh_range, flag = flag, depth=depth+1)

    if flag != 1:
        if best_params[1] == thresh_range[0]:
            print('thresh too low', best_params[1], thresh_range)
            delta = thresh_range[1]-thresh_range[0]
            new_min = thresh_range[0]-5*delta
            new_max = thresh_range[0]+1*delta
            new_range = range(new_min, new_max, delta)
            #_pdb.set_trace()
            best_params, _ = loop_params(cell_id_or_args, peri_weight_range, new_range, flag = flag, depth=depth+1)
        elif best_params[1] == thresh_range[-1]:
            print('thresh too hi', best_params[1], thresh_range)
            delta = thresh_range[1]-thresh_range[0]
            new_min = thresh_range[-1]-1*delta
            new_max = thresh_range[-1]+5*delta
            new_range = range(new_min, new_max, delta)
            #_pdb.set_trace()
            best_params, _ = loop_params(cell_id_or_args, peri_weight_range, new_range, flag = flag, depth=depth+1)

    # report param value for which error is minimized
    print('Minimum parameters are: {0}'.format(best_params))
    
    _plt.close('errors')
    fig, ax = _plt.subplots(num='errors')
    ax.plot(error)

    return best_params, args

def fit_cells(cell_id):
    '''
    Find best parameters for both peripheral_weight and nl_thresh
    I loop over both parameters very coarsley, pick best set
    Then Loop over threshold only, more finely -> pick best threshold
    Then loop over periphery_weight more finely -> pick best
    Then over threshold again
    
    input:
    ------
        cell_id:        iterable of ints or int

    output:
        best_params_dict:       dictionary with the best parameters associated with a given cell
                                result is also written to 'UFlicker PSTHs/best_parameters.txt'
    '''

    
    #_pdb.set_trace()

    print(cell_id)
    if not _np.iterable(cell_id):
        cell_id = [cell_id]

    best_params_dict = {}

    best_params_to_file(['cell_id', 'peri_weight', 'nl_thresh', 'error'])

    #_pdb.set_trace()
    for cell in cell_id:
        print('working on cell {0}'.format(cell))
        # coarse search
        result0 = loop_params(cell, range(20, 100, 20), range(20, 120, 20))
        if result0 is None:
            continue
        
        best_params, args = result0 
        print(best_params)

        # fix 1st param and search 2nd more finely
        best_params, _ = loop_params(args, range(best_params[0], best_params[0]+1, 1), range(best_params[1]-10, best_params[1]+10, 2), flag=2)
        print(best_params)

        # fix 2nd and search 1st
        best_params, _ = loop_params(args, range(best_params[0]-10, best_params[0]+10, 2), range(best_params[1], best_params[1]+1, 1), flag=1)
        print(best_params)

        # fix 1st again and search 2nd
        best_params, _ = loop_params(args, range(best_params[0], best_params[0]+1, 1), range(best_params[1]-10, best_params[1]+10, 2), flag=2)
        print(best_params)

        best_params_dict[cell]=best_params
        
        best_params_to_file((cell,) + best_params)
    
    return best_params_dict

def best_params_to_file(val_list):
    '''
    write best_params_dict to file
    '''

    #_pdb.set_trace()
    with open('UFlicker PSTHs/best_parameters.txt', 'at') as fid:
        string = '{0} {1} {2} {3}\n'.format(val_list[0], val_list[1], val_list[2], val_list[3])
        #_pdb.set_trace()
        fid.write(string)

def average_best_params():
    '''
    after running fit_cells() for all cells, file 'UFlicker PSTHs/best_parameters.txt' was created. Each line in file has the cell_id, peripheral_weight and nl_thresh for the cell.
    Compute average and sd for both parameters.
    '''

    df = _pd.read_csv('UFlicker PSTHs/best_parameters.txt', sep=' ')
    
    peri_weight = df['peri_weight']
    nl_thresh = df['nl_thresh']

    return peri_weight.mean(), peri_weight.std(), nl_thresh.mean(), nl_thresh.std() 
