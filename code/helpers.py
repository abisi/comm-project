import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

# ---- DATA PREPROCESSING ---- #

#def convolve_1d(x, axis, kernel_size=5):
def convolve_1d(x, kernel_size=5):
    
    kernel = np.ones(kernel_size) / kernel_size
    x_convolved  = np.convolve(x, kernel, mode='same')
    #x_convolved = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='full'), axis=axis, arr=x)
    return x_convolved

def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x

def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)

def slice_array(spike_array, pre_start_bins, post_start_bins, smooth=True, smooth_kernel=5):
    """
    Slice spike array

    Parameters
    ----------
    spike_array : Neurons x trials x time_bins
        spike count matrix
    pre_start_bins : number of bins before trial start
    post_start_bins : number of bins before trial start
    smooth: whether to smooth
    smooth_kernel: window size in (in bins; 1 bin=10ms)
    
    Returns
    -------

    spike_array_processed
    -> slices spike array
    """
    n_neurons = spike_array.shape[0]
    n_trials = spike_array.shape[1]
    spike_array_processed = []

    trial_start_idx = 200

    start_idx = trial_start_idx - pre_start_bins - smooth_kernel # to take border artefact into account
    stop_idx =  trial_start_idx + post_start_bins + smooth_kernel
    
    for neur_idx in range(n_neurons):
        spike_array_trial = []
        for t_idx in range(n_trials):
            trial_spikes = spike_array[neur_idx, t_idx, start_idx : stop_idx]            
            
            if smooth:
                trial_spikes = convolve_1d(trial_spikes, kernel_size=smooth_kernel)
                
            spike_array_trial.append(trial_spikes[smooth_kernel:-smooth_kernel])
        spike_array_processed.append(spike_array_trial)    
    
    return np.asarray(spike_array_processed)


def split_reshape(X):
    """
    Split into train-test and reshape 2D
    X: neurons x trials x time bins
    """
    
    print(X.shape)
    n_trials = X.shape[1]
    print(n_trials)
    if n_trials % 2 == 1:
        n_trials -= 1
    
    print(n_trials)
    # Shuffle-split
    idx = np.arange(n_trials)
    np.random.shuffle(idx)
    X = X[:,idx,:]
    X_tr, X_te = X[:,n_trials//2:,:], X[:,:n_trials//2,:]
    
    # Reshape
    
    X_tr = X_tr.reshape(X_tr.shape[0], -1)
    X_te = X_te.reshape(X_te.shape[0], -1)

    return X_tr, X_te

# ---- PLOTTING  ---- #

def remove_top_right_frame(ax):
    ax.spines[['right', 'top']].set_visible(False)
    return


def get_trial_type_color(trial_type):
    """
    Get trial type color.
    """
    trial_colors = {'ah':'mediumblue',
               'wh':'forestgreen',
               'wm':'crimson',
               'fa':'k',
               'cr':'dimgray'}
    return trial_colors[trial_type]


def get_area_color(area):
    """
    Get area color.
    """
    area_colors = {
                   'DLS':'mediumorchid',
                   'DS':'plum',
                   'tjM1':'peru',
                   'OFC':'darkslateblue',
                   'wS1':'crimson',
                   'wM2':'seagreen',
                   'Thalamus':'salmon',
                   'nS1':'slategray',
    }
    return area_colors[area]

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def return_timebin_cmap(color, n_bins):
    """
    Returns n-bin colormap.
    """
    colors = [lighten_color(color, amount=n) for n in np.linspace(0.2,1.0,n_bins)]
    return colors

