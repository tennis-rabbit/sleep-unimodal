# class for EEG spectrogram generation 
# combine known preprocessing steps, Yuzhe's prior code, and various methods for spectrogram generation 
"""
Three sources: 
https://olebialas.github.io/posts/eeg_preprocessing/
https://prerau.bwh.harvard.edu/multitaper/
Yuzhe's previous preprocessing code 
My experience and reading around the web


Things to note: 

1. types of methods (toggle with params):
- classic STFT 
- welch's 
    - is there a difference between small window FFT + moving average and welch's method across time?
    - these are essentially the same but small window FFT + moving average is faster because FFT is computed for each window once
- classifc STFT followed by moving average across time
- multitaper spectral analysis 
     - fine time-frequency representation with mne.time_frequency.tfr_array_multitaper
     - short time multitaper spectral analysis applying mne.time_frequency.psd_array_multitaper across windows

2. preprocessing (toggle with params): 
- output mode type (psd vs magnitude) 
- linear and non-linear detrending 
- power line noise filtering
- shift and scale to match another dataset
- artifact removal? (do not implement yet, but note that ICA can be used to identify artifacts (eeglab_detect_artifact.py) or template based artifact removal (eeg_artifact_remover.py))
- Simpson's rule integration to define power in each EEG band

3. output: 
- psd 
- eeg bands if toggled on
"""

import numpy as np
import pandas as pd
from scipy.signal import spectrogram, welch, savgol_filter, convolve2d
from meegkit.dss import dss_line # remove powerline noise
from scipy import integrate
from mne.time_frequency import tfr_array_multitaper, psd_array_multitaper
import matplotlib.pyplot as plt 

EEG_BANDS_DICT = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gamma": (30.0, 64) # 64, assuming 128 Hz sampling rate for EEG as we discussed
    }

class EEGSpec:
    """
    Class for EEG spectrogram generation

    Multiple methods are supported, see the following functions: 
        {'stft', 'stft_average', 'welch', 'multitaper'}

    Assumes that signal is already low-pass filtered at Nyquist frequency and normalized (locally)
    """
    def __init__(self, signal, fs, output_mode = 'density', eeg_bands = True, detrend = True, powerline_freq = None, shift_scale = None):
        """
        Initialize the EEGSpec object

        Params:
            signal: numpy array of EEG signal 
            fs: sampling rate of EEG signal (e.g. 128)
            output_mode: units of output, one option from: ['density'(power density), 'spectrum'(power)]
            eeg_bands: (T/F) whether to compute eeg bands
            detrend: (T/F) whether to detrend the signal
            powerline_freq: frequency of powerline noise, ignore if None
            shift_scale: {'mean': float, 'std': float} or None if ignore
        """

        print(f"Initializing EEGSpec object")

        self.signal = signal
        self.fs = fs
        self.output_mode = output_mode
        self.eeg_bands = eeg_bands
        self.detrend = detrend
        self.powerline_freq = powerline_freq
        self.shift_scale = shift_scale

        self.signal_processed = None
        self._setup()

    def _setup(self):
        """
        Run preprocessing steps
        """

        print("Preprocessing signal")
        signal_processing = self.signal

        if self.shift_scale is not None:
            # for when you want to shift and scale to match another dataset
            print(f"Shifting and scaling signal")
            # assuming signal is locally centered and scaled but not globally
            signal_processing = (signal_processing - np.mean(signal_processing)) / np.std(signal_processing)
            signal_processing = signal_processing * self.shift_scale['std'] + self.shift_scale['mean']

        if self.detrend:
            print("Detrending signal")
            window_length = int(self.fs * 15) * 2 + 1 # it must be an odd integer, the value at the center of the window is replaced with value of fitted polynomial
            signal_processing = signal_processing - savgol_filter(signal_processing, window_length, polyorder=2)

        if self.powerline_freq is not None:
            print(f"Removing powerline frequency {self.powerline_freq} Hz")
            if self.powerline_freq <= self.fs / 2:
                signal_processing_reshape = signal_processing[:,np.newaxis] # (T, 1)
                signal_processing_reshape, noise = dss_line(signal_processing_reshape, sfreq=self.fs, fline=self.powerline_freq)
                signal_processing = signal_processing_reshape.ravel() # (T,)
            else:
                print(f"Powerline frequency {self.powerline_freq} Hz is greater than half the sampling rate {self.fs / 2} Hz, ignoring")

        self.signal_processed = signal_processing

    def compute_band_power(self, spectrogram, freqs):
        """
        Given the 2D spectrogram, compute the power in each EEG band using Simpson's rule for integration and EEG_BANDS_DICT definitions of bands

        Params: 
            spectrogram: 2D numpy array (F, T)
            freqs: 1D numpy array (F,) of frequencies in spectrogram 
            
        Returns:
            band_power: dict mapping band names to 1D numpy arrays (T,) containing power in that band over time
        """
        band_power = {}
        
        for band_name, (fmin, fmax) in EEG_BANDS_DICT.items():
            # Get indices for frequencies in this band
            band_mask = (freqs > fmin) & (freqs <= fmax)
            
            # Integrate power over frequencies in band at each timepoint
            power = integrate.simpson(y = spectrogram[band_mask], x = freqs[band_mask], axis=0)
            band_power[band_name] = power
            
        return band_power

    def stft(self, win_sec, step_sec, window='hann', pad = True):
        """
        Compute STFT of signal.
        This is just a classic STFT, where FFT is applied to each window across time. 

        Params: 
            win_sec: window size in seconds 
            step_sec: step size in seconds
            window: type of window tapering, see scipy documentation
            pad: (T/F) whether to pad the signal to get better resolution at start and end
        """
        
        print('Running classic STFT')

        # get processed signal
        signal = self.signal_processed

        # max frequency to consider
        max_freq = self.fs / 2

        # pad to get better resolution at start and end
        if pad:
            pad_length = self.fs * ((win_sec - step_sec) // 2) 
            signal = np.concatenate((np.zeros(pad_length, dtype=np.float64),
                                      signal, 
                                      np.zeros(pad_length, dtype=np.float64)), axis=0)
        
        # define window size and step size and compute spectrogram
        nperseg = int(win_sec * self.fs)
        step = int(step_sec * self.fs)
        freqs, times, Sxx = spectrogram(x=signal, fs=self.fs, 
                                        window=window, nperseg=nperseg,
                                        noverlap=nperseg - step, detrend='linear', 
                                        scaling=self.output_mode) # times does not align with original times 
        
        # re-adjust time to original 
        if pad:
            pad_sec = pad_length / self.fs
            times = times - pad_sec


        # keep relevant frequencies 
        keep = freqs <= max_freq
        freqs = freqs[keep]
        Sxx = Sxx[keep, :]

        if self.eeg_bands:
            band_power = self.compute_band_power(Sxx, freqs)
            return freqs, times, Sxx, band_power
        else:
            return freqs, times, Sxx
        
    def stft_average(self, avg_win_num, stft_win_sec, stft_step_sec, window='hann', pad = True):
        """
        Compute STFT of signal with small windows, and apply a moving average across neighboring windows.
        This is equivalent to Welch's method. This might be faster because FFT is computed for each window once.
        Suggestion: set stft_win_sec small 
        
        Params: 
            avg_win_num: number of windows to average over
            stft_win_sec: window size for STFT in seconds
            stft_step_sec: step size for STFT in seconds
            window: type of window tapering, see scipy documentation
            pad: (T/F) whether to pad the signal to get better resolution at start and end
        """
        
        print('Running STFT with moving average')

        # just run stft function to get spectrogram 
        freqs, times, Sxx, _ = self.stft(stft_win_sec, stft_step_sec, window, pad)

        # apply moving average across windows 
        Sxx_smooth = convolve2d(Sxx, np.ones((1, avg_win_num))/avg_win_num, mode='valid') # temporal smoothing with convolution w 1 x K kernal
        
        # edit times to reflect new time resolution after moving average        
        times_smooth = np.convolve(times, np.ones(avg_win_num) / avg_win_num, mode='valid') # 1D convolution of times with kernel of ones

        if self.eeg_bands:
            band_power = self.compute_band_power(Sxx_smooth, freqs)
            return freqs, times_smooth, Sxx_smooth, band_power
        else:
            return freqs, times_smooth, Sxx_smooth
        
    def welch(self, frame_win_sec, frame_step_sec, welch_win_sec, welch_step_sec, window='hann', pad = True):
        """
        Apply Welch's method repeatedly on sliding frame. For each frame, Welch's algorithm further splits into smaller segments and averages FFT results. 

        Params: 
            frame_win_sec: window size for each frame in seconds
            frame_step_sec: step size for each frame in seconds
            welch_win_sec: window size for Welch's method in seconds
            welch_step_sec: step size for Welch's method in seconds
            window: type of window tapering, see scipy documentation
            pad: (T/F) whether to pad the signal to get better resolution at start and end
        """
        
        print('Running Welch')

        # get processed signal
        signal = self.signal_processed

        # max frequency to consider
        max_freq = self.fs / 2

        # pad to get better resolution at start and end 
        if pad:
            pad_length = self.fs * ((frame_win_sec - frame_step_sec) // 2) 
            signal = np.concatenate((np.zeros(pad_length, dtype=np.float64),
                                      signal, 
                                      np.zeros(pad_length, dtype=np.float64)), axis=0)
            
        # define welch window size and step size 
        nperseg = int(welch_win_sec * self.fs)
        step = int(welch_step_sec * self.fs)

        # for each frame, compute welch's method and store results 
        nperframe = int(frame_win_sec * self.fs)
        step_frame = int(frame_step_sec * self.fs)
        nframe = (len(signal) - nperframe) // step_frame + 1 # total number of frames
        psd_list = []
        for i in range(nframe):
            seg = signal[i*step_frame : i*step_frame + nperframe] # segment of interest
            freqs, psd = welch(seg, fs=self.fs, window=window, nperseg=nperseg, 
                           noverlap=nperseg - step, detrend = 'linear', 
                           scaling = self.output_mode) # welch returns freqs and psd
            psd_list.append(psd)
        Sxx = np.asarray(psd_list).T # (F, T)
        times = np.arange(nframe) * frame_step_sec + (frame_win_sec / 2) # center of frame
        
        if pad:
            pad_sec = pad_length / self.fs
            times = times - pad_sec

        # keep relevant frequencies 
        keep = freqs <= max_freq
        freqs = freqs[keep]
        Sxx = Sxx[keep, :]

        if self.eeg_bands:
            band_power = self.compute_band_power(Sxx, freqs)
            return freqs, times, Sxx, band_power
        else:
            return freqs, times, Sxx

    def multitaper(self, win_sec, time_bandwidth=4, pad = True):
        """
        NOTE: This function is slow because tfr_array_multitaper computes power across frequencies for every timepoint (using window centered at that timepoint). This does not allow for flexibility of changing window size and step size. 
        WARNING: This function needs further testing. I implemented this but it was too slow to test on entire night of data.
        TO-DO: Change implementation to manually slide a window across time like Welch's above, and use psd_array_multitaper
        
        
        Apply multitaper spectral analysis to signal. See https://mne.tools/stable/generated/mne.time_frequency.tfr_array_multitaper.html

        Params: 
            win_sec: window size in seconds 
            time_bandwidth: time bandwidth product, controls the number of tapers
            pad: (T/F) whether to pad the signal to get better resolution at start and end

        time_bandwidth is a critical parameter because it controls the number of tapers as floor(time_bandwidth - 1)
        frequency_resolution = time_bandwidth/win_sec
        freqs_interest =  np.arange(min, max, frequency_resolution)
        n_cycles = freqs_interest*win_sec
        win_sec = n_cycles / freqs_interest
        """
        
        print('Running Multitaper spectral analysis')

        # get processed signal
        signal = self.signal_processed

        # max frequency to consider
        max_freq = self.fs / 2
        # min frequency to consider 
        min_freq = 0.5

        # pad to get better resolution at start and end
        if pad:
            pad_length = self.fs * (win_sec // 2) 
            signal = np.concatenate((np.zeros(pad_length, dtype=np.float64),
                                      signal, 
                                      np.zeros(pad_length, dtype=np.float64)), axis=0)

        # compute parameters for multitaper spectral analysis 
        frequency_resolution = time_bandwidth/win_sec
        freqs_interest =  np.arange(min_freq, max_freq, frequency_resolution)
        n_cycles = freqs_interest*win_sec

        # compute multitaper spectral analysis 
        data = signal[np.newaxis, np.newaxis, :]
        tfr = tfr_array_multitaper(data, sfreq=self.fs, freqs=freqs_interest, n_cycles=n_cycles,
                           time_bandwidth=time_bandwidth, output='power')

        if self.output_mode == 'density':
            Sxx = tfr.squeeze()/frequency_resolution  # shape: (n_freqs, n_times)
            # divide by frequency resolution to get power density in units V^2/Hz
        else:
            Sxx = tfr.squeeze() # otherwise power in units V^2

        # clip off frequency representations as padded values
        Sxx = Sxx[:,pad_length:Sxx.shape[1]-pad_length]

        if self.eeg_bands:
            band_power = self.compute_band_power(Sxx, freqs_interest)
            return freqs_interest, Sxx, band_power
        else:
            return freqs_interest, Sxx

    # can also implement a method that repeatedly applies psd_array_multitaper across windows of time 
    # tfr_array_multitaper is slow because it get's frequency representation for every input timepoint 
        # centers a time window at each timepoint and applies FFT with multiple tapers and averages across tapers 

    def multitaper_sliding_window(self, win_sec, step_sec, bandwidth=1, pad = True):
        """
        Apply multitaper spectral analysis to signal with sliding time windows. 
        see: https://mne.tools/stable/generated/mne.time_frequency.psd_array_multitaper.html
            this multitaper spectral analysis method does not give you time-frequency representation, but instead give you the frequency representation of your entire input length
            so we use this on sliding windows of time

        Params:
            win_sec: window size in seconds for which to apply each multitaper spectral analysis
            step_sec: step size in seconds for which to slide the window
            bandwidth: half-frequency bandwidth, indirectly controls the number of tapers (see bottom of https://mne.tools/stable/generated/mne.time_frequency.tfr_array_multitaper.html)

        bandwidth = half-frequency bandwidth
        number of tapers = (2*bandwidth*win_sec) - 1
        """

        print("Running Multitaper spectral analysis with sliding time windows")

        # get processed signal
        signal = self.signal_processed

        # max frequency to consider
        max_freq = self.fs / 2

        # min frequency to consider 
        min_freq = 0.5

        # pad to get better resolution at start and end
        if pad:
            pad_length = self.fs * (win_sec // 2) 
            signal = np.concatenate((np.zeros(pad_length, dtype=np.float64),
                                      signal, 
                                      np.zeros(pad_length, dtype=np.float64)), axis=0)

        # for each frame, compute multitaper method and store results 
        nperseg = int(win_sec * self.fs)
        step = int(step_sec * self.fs)
        nframe = (len(signal) - nperseg) // step + 1 # total number of frames
        psd_list = []
        for i in range(nframe):
            seg = signal[i*step : i*step + nperseg] # segment of interest
            psd, freqs = psd_array_multitaper(seg, sfreq=self.fs, fmin=min_freq, fmax=max_freq, 
                            bandwidth=bandwidth, remove_dc=True, normalization='length', output='power', verbose=False)
            psd_list.append(psd)
        Sxx = np.asarray(psd_list).T # (F, T)
        times = np.arange(nframe) * step_sec + (win_sec / 2) # center of frame

        # if you want power in units V^2, you need to multiple by the frequency spacing
        if self.output_mode == 'spectrum':
            delta_f = freqs[1] - freqs[0]
            Sxx = Sxx * delta_f
        
        # adjust time if padding was used
        if pad:
            pad_sec = pad_length / self.fs
            times = times - pad_sec

        # keep relevant frequencies 
        keep = freqs <= max_freq
        freqs = freqs[keep]
        Sxx = Sxx[keep, :]

        if self.eeg_bands:
            band_power = self.compute_band_power(Sxx, freqs)
            return freqs, times, Sxx, band_power
        else:
            return freqs, times, Sxx


# --- plotting code here
def plot_eeg_band_post_spectrogram_with_hypnogram(freqs, times, Sxx, stages, sfreq, start_t, log_scale=True,
                                                 max_freq_plot=32, epoch=30, stage_labels=None, cmap='jet', 
                                                 stage_order=['Unknown', 'Awake', 'REM', 'Light Sleep', 'Deep Sleep']):
    """
    Plots EEG spectrogram with aligned sleep stage hypnogram.
    ASSUMES SLEEP STAGE DESCRIBE THE PREVIOUS WINDOW AND STARTS AT epoch

    Parameters:
    - freqs, times, Sxx: result of spectrogram generation; frequency bins, time bins, and FxT spectrogram
    - stages (list): list of sleep stage codes with indices as time (one stage per epoch)
    - sfreq (float): Sampling frequency of signal
    - start_t (float): Start time if not starting at t = 0 (important for plotting alignment)
    - log_scale (bool): whether or not to plot on the log scale
    - max_freq_plot (int): maximum frequency to plot
    - epoch (int): Epoch length of sleep stages in seconds
    - stage_labels (dict): Optional mapping of stage codes to human-readable names
    - cmap (str): Colormap for the spectrogram
    - stage_order: Order of hypnogram y axis from bottom to up
    """

    # Step 1: Convert Sxx to dB scale
    if log_scale:
        Sxx = 10 * np.log10(Sxx + 1e-10)
        label = 'Power (dB/Hz)'
    else:
        label = 'Power (V*2/Hz)'
    print(Sxx.max(), Sxx.min())
    # Step 2: Subset frequencies and adjust start time
    freq_mask = freqs <= max_freq_plot
    freqs = freqs[freq_mask]
    Sxx = Sxx[freq_mask, :]
    times = times + start_t

    # Step 3: Convert stage codes to labels
    if stage_labels is None:
        # Default label mapping
        stage_labels = {
            -1: 'Unknown',
            0: 'Awake',
            1: 'Light Sleep',
            2: 'Light Sleep',
            3: 'Deep Sleep',
            4: 'REM'
        }
    # Convert stages to names
    stages = stages.loc[:max(times)]
    stage_names = [stage_labels.get(s, 'Unknown') for s in stages]

    # Step 4: Map labels to y-values for hypnogram
    label_to_yval = {label: i for i, label in enumerate(stage_order)}
    y_vals = [label_to_yval[name] for name in stage_names]

    # Step 5: Build stage time axis (assumes stages mark end of epochs)
    stage_times = (np.arange(len(y_vals)) * epoch) + start_t

    # Step 6: Plot spectrogram and hypnogram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Spectrogram
    t_mesh, f_mesh = np.meshgrid(times, freqs)
    im = ax1.pcolormesh(t_mesh, f_mesh, Sxx, cmap=cmap)
    ax1.set_ylabel('Frequency (Hz)', fontsize=12)
    ax1.set_title('EEG Spectrogram', fontsize=14)
    fig.colorbar(im, ax=[ax1, ax2], label=label)

    # Hypnogram
    ax2.step(stage_times, y_vals, where='post', color='blue')
    ax2.set_yticks(list(label_to_yval.values()))
    ax2.set_yticklabels(stage_order)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Sleep Stage', fontsize=12)
    ax2.set_title('Hypnogram', fontsize=14)

    #plt.tight_layout()
    ax1.set_xlim(left=min(times[0], stage_times[0]))
    plt.show()
    
    return
# ---


# --- local normalization code from Yuzhe -> incorporate into preprocessing 
from scipy.signal import convolve

def compute_local_std_mean0(length, input_data):
    # local normalization
    assert length % 2 == 0
    # speeding up algorithm
    ave_kernel = np.ones((length,), dtype='float32') / length
    local_mean = convolve(input_data, ave_kernel, mode='same')
    residual = input_data - local_mean
    residual_square = residual ** 2 # residual square
    local_std = convolve(residual_square, ave_kernel, mode='same') ** 0.5 + 1e-30 # variance to standard deviation 
    return np.divide(residual, local_std) # normalize by standard deviation
# ---