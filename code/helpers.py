import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import pandas as pd

# ---- DATA LOADING ---- #

#trial_info classs
class trial_info:
    def __init__(self, datapath):
        self.data = pd.read_csv(datapath)
        self.trial_types = ["Auditory Hit", "Auditory Miss","Whisker Hit", 
                            "Whisker Miss", "False Alarm", "Correct Rejection"]
        self._trial_index = None
        self._num_trials = None
        
    @property
    def trial_index(self):
        ah_idx = np.where(self.data["ah"] == 1)[0]
        am_idx = np.where(self.data["am"] == 1)[0]
        wh_idx = np.where(self.data["wh"] == 1)[0]
        wm_idx = np.where(self.data["wm"] == 1)[0]
        fa_idx = np.where(self.data["fa"] == 1)[0]
        cr_idx = np.where(self.data["cr"] == 1)[0]
        
        all_idx = [ah_idx, am_idx, wh_idx, wm_idx, fa_idx, cr_idx]
        
        self._trial_index = {self.trial_types[i]: all_idx[i] for i in range(len(self.trial_types))}
        
        return self._trial_index
    
    @property
    def num_trials(self):
        if self._trial_index == None:
            _ = self.trial_index
            
        self._num_trials = {key: len(self._trial_index[key]) for key in self._trial_index}
            
        return self._num_trials

#neuron_info classs
class neuron_info:
    def __init__(self, datapath):
        self.data = pd.read_csv(datapath)
        self.n_neurons = self.data.shape[0]
        self.areas = self.data["area"].unique()
        self._area_idx = None
        self._n_neurons_area = None
    
    @property
    def area_idx(self):
        self._area_idx = {area: np.where(self.data["area"] == area)[0] for area in self.areas}
        
        return self._area_idx
    
    @property
    def n_neurons_area(self):
        if self._area_idx == None:
            _, self.area_idx
        
        self._n_neurons_area = {area: len(self.area_idx[area]) for area in self.area_idx}
        
        return self._n_neurons_area
    
def filter_data(spike_bins, trial_inf, neuron_inf, trial_types, areas):
    if not isinstance(trial_inf, trial_info):
        raise ValueError(f"Expected trial inf to be of type __main__.trial_info but got {type(trial_inf)} instead")
    
    if not isinstance(neuron_inf, neuron_info):
        raise ValueError(f"Expected trial inf to be of type __main__.neuron_info but got {type(trial_inf)} instead")
        
    if not isinstance(areas, list):
        raise ValueError(f"areas to be of type list but got {type(areas)} instead")
        
    if not isinstance(trial_types, list):
        raise ValueError(f"trial_types to be of type list but got {type(areas)} instead")

        
    f_data = {}
    for trial_type in trial_types:
        if trial_type not in trial_inf.trial_types:
            raise NameError(f"Trial type can only be one of the following: {trial_inf.trial_types}")
        
        else:
            f_data[trial_type] = {}
            trial_type_idx = trial_inf.trial_index[trial_type]
            for area in areas:
                if area not in neuron_inf.areas:
                    print(area)
                    raise NameError(f"Area can only be one of the following: {neuron_inf.areas}")
                    
                else:
                    area_idx = neuron_inf.area_idx[area]
                    
                    
                    f_data[trial_type][area] = spike_bins[area_idx][:,trial_type_idx,:]
                    
    return f_data

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

    start_idx = trial_start_idx - pre_start_bins
    stop_idx =  trial_start_idx + post_start_bins
    
    for neur_idx in range(n_neurons):
        spike_array_trial = []
        for t_idx in range(n_trials):
            trial_spikes = spike_array[neur_idx, t_idx, start_idx : stop_idx]            
            
            if smooth:
                trial_spikes = convolve_1d(trial_spikes, kernel_size=smooth_kernel)
                
            spike_array_trial.append(trial_spikes)
        spike_array_processed.append(spike_array_trial)    
    
    return np.asarray(spike_array_processed)

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

