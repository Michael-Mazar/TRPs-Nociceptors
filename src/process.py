import scipy as sp
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, detrend
import matplotlib.pyplot as plt
import pdb 

def down_sample(vtrace, ttrace, tstop, dt):
    """Down sample data by averaging over a window of size down_sample_rate."""
    downsample = int(1/dt) # the goal down sample so each point is 1 ms
    v_downV = sp.signal.resample(vtrace,\
                            int((len(vtrace)/downsample)), window='hamm')
    t_downt = ttrace[::int(downsample)]
    t_downt = t_downt[:-1]

    # Verify downsampling occured to 1 ms
    assert tstop == len(v_downV), pdb.set_trace()# "Downsampling did not occur to 1 ms", 
    return v_downV, t_downt

def crop_trace(vtrace, ttrace, delay, duration, dt, edge_window=50):
    """
    Crop the voltage and time traces to the new duration.
    edge_window: The window to crop the trace to. usually after first action potential
    """
    start_stim = int((delay - edge_window)/dt) # Divide dt to fix and get the right indices
    end_stim = int((duration + edge_window)/dt) #TODO: Add edge window (?)
    stimulated_trace = vtrace[start_stim:end_stim]
    stimulated_time = ttrace[start_stim:end_stim]
    return stimulated_trace, stimulated_time

def detrend_trace(vtrace, degree, type = 'constant'):
    """
    Detrend the trace by cropping and removing the mean
    """
    p = np.polyfit(np.arange(len(vtrace)), vtrace, degree) # Fit a polynomial to the data
    trend = np.polyval(p, np.arange(len(vtrace))) # Evaluate the polynomial
    v_detrended_ramp = vtrace - trend # Detrend the signal
    return v_detrended_ramp

def smooth_trace(vtrace, type='savgol'):
    """
    Smooth the trace using a Savitzky-Golay filter
    """
    if type == 'savgol':
        v_smooth = sp.signal.savgol_filter(vtrace, 11, 3) # Savitzky-Golay filter
    return v_smooth

def normalize_waveforms(waveform_data):
    # Z-score Normalization
    mean = np.mean(waveform_data)
    std_dev = np.std(waveform_data)
    normalized_waveform = (waveform_data - mean) / std_dev
    return normalized_waveform

def normalize_currents(C, positive=True):
    # Isolate positive/negative currents
    C_filtered = np.where(C > 0, C, 0) if positive else np.where(C < 0, C, 0)  # Note the negation for negative currents

    # Calculate normalization vector
    n_filtered = C_filtered.sum(axis=0)

    # Normalize the currents
    C_hat_filtered = np.divide(C_filtered, n_filtered, where=n_filtered!=0)  # Avoid division by zero

    # Generate the Currentscapes Matrix
    R = 2000  # Resolution factor
    C_S = np.zeros((R, C.shape[1]))

    for j in range(C_S.shape[1]):
        p_ij = C_hat_filtered[:, j] * R
        cum_p_ij = np.cumsum(p_ij)
        for i in range(R):
            for k in range(len(p_ij)):
                if i < cum_p_ij[k] and (k == 0 or i >= cum_p_ij[k-1]):
                    C_S[i, j] = k
                    break
    
    return C_S, n_filtered

def get_stacked_currents(data_dict, currents_names):
    """
    Stack the currents for the underlying currents.
    """
    stacked_data = np.vstack(tuple([data_dict[key] for key in currents_names]))
    return stacked_data

def get_firing_threshold(voltage, time):
    # Compute the derivative of the voltage with respect to time (dV/dt)
    dv_dt = np.diff(voltage) / np.diff(time)
    
    # Find max and min values of dV/dt
    dv_dt_max = np.max(dv_dt)
    dv_dt_min = np.min(dv_dt)
    
    # Calculate the firing threshold value
    # threshold_value = 0.03 * (dv_dt_max - dv_dt_min) + dv_dt_min
    threshold_value = 0.5 * (dv_dt_max - dv_dt_min) + dv_dt_min
    
    # Find the index where dV/dt reaches the threshold
    threshold_index = np.where(dv_dt >= threshold_value)[0][0]
    
    # Return the time and voltage at the threshold
    threshold_time = time[threshold_index]
    threshold_voltage = voltage[threshold_index]

    return threshold_voltage
    # return threshold_voltage, threshold_time, threshold_voltage, threshold_value

def extract_eletrophysiological_features(label, trace_df, ap_data_frame, trace_dict):
    """
    Setup an electrophysiological feature extraction for the different conditions

    """
    df = trace_df.copy()
    df['Run'] = (df['run'].astype(int))  # Fix run column #TODO: Make it fixed before hand

    # Extract current threshold
    df_filtered = df[df['Spikecount'] == 1] # Filter the rows where Spikecount is 1
    df_sorted = df_filtered.sort_values(by=['Run', 'current'], ascending=[True, True]) # Sort the dataframe by 'Run' and 'current'
    df_grouped = df_sorted.groupby('Run').first().reset_index() # Step 4: Group by 'Run' and extract the first row with the minimal current
    # Select the relevant columns
    columns_to_keep = [ 
        'Run', 'current', 'identifier'
    ]
    df_final = df_grouped[columns_to_keep]
    df_final.rename(columns={'current': 'Current Threshold'}, inplace=True)

    # Add the condition name for the runs 
    df_final.insert(0, 'Condition', label)

    # Extract the AP threshold for each of the identifiers in df_final using function and traces
    df_final['AP Threshold'] = df_final.apply(lambda row: get_firing_threshold(
        trace_dict[str(row['identifier'])]['voltage'],
        trace_dict[str(row['identifier'])]['time']), axis=1) 

    # Get the relevant AP features
    relevant_features = ['peak_time', 'AP_duration', 'AP_height']
    df_final = df_final.merge(ap_data_frame[relevant_features + ['identifier']], on='identifier', how='left')

    # Get voltage base feature from data for each run
    voltage_base_df = df[df['current'] == 0.0][['Run', 'voltage_base']]
    df_final = df_final.merge(voltage_base_df, on='Run', how='left')

    return df_final

def get_currentscape(voltage, currents):
    # Adapted from https://github.com/leandro-alonso/homeostasis/blob/main/currents_visualization.py
    curr=np.array(currents)	
    cpos= curr.copy()
    cpos[curr<0]=0
    cneg= curr.copy()
    cneg[curr>0]=0

    normapos = np.sum(abs(np.array(cpos)),axis=0)
    normaneg = np.sum(abs(np.array(cneg)),axis=0)
    npPD=normapos
    nnPD=normaneg
    cnorm=curr.copy()
    cnorm[curr>0]=(abs(curr)/normapos)[curr>0]
    cnorm[curr<0]=-(abs(curr)/normaneg)[curr<0]

    resy=1000
    impos=np.zeros((resy,np.shape(cnorm)[-1])) 
    imneg=np.zeros((resy,np.shape(cnorm)[-1])) 

    times=np.arange(0,np.shape(cnorm)[-1])
    for t in times:
        lastpercent=0
        for numcurr, curr in enumerate(cnorm):
            if(curr[t]>0):
                percent = int(curr[t]*(resy))   
                impos[lastpercent:lastpercent+percent,t]=numcurr
                lastpercent=lastpercent+percent        
    for t in times:
        lastpercent=0
        for numcurr, curr in enumerate(cnorm):
            if(curr[t]<0):
                percent = int(abs(curr[t])*(resy))   
                imneg[lastpercent:lastpercent+percent,t]=numcurr
                lastpercent=lastpercent+percent        
    image = np.vstack((impos,imneg))
    return image, npPD, nnPD



def process_currentspace(C, positive=True, R = 2000):
            # Isolate positive/negative currents
            C_filtered = np.where(C > 0, C, 0) if positive else np.where(C < 0, C, 0)  # Note the negation for negative currents

            # Calculate normalization vector
            n_filtered = C_filtered.sum(axis=0)

            # Normalize the currents
            C_hat_filtered = np.divide(C_filtered, n_filtered, where=n_filtered!=0)  # Avoid division by zero

            # Generate the Currentscapes Matrix
            C_S = np.zeros((R, C.shape[1]))

            for j in range(C_S.shape[1]):
                p_ij = C_hat_filtered[:, j] * R
                cum_p_ij = np.cumsum(p_ij)
                for i in range(R):
                    for k in range(len(p_ij)):
                        if i < cum_p_ij[k] and (k == 0 or i >= cum_p_ij[k-1]):
                            C_S[i, j] = k
                            break
            
            return C_S, n_filtered
def save_cell_setting_to_json(self, cell, filename):
    print("Saving to json")
    
        
# Apply bandpass filter to the membrane voltage data
def butter_bandpass(self, lowcut, highcut, fs, order=4):
    """
    Design a Butterworth bandpass filter.
        - lowcut: Lower cutoff frequency (Hz)
        - highcut: Upper cutoff frequency (Hz)
        - fs: Sampling rate of the data (Hz)
        - order: Order of the filter (default = 4)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(self, data, lowcut, highcut, fs, order=4):
    b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def _butter_lowpass(self, cutoff, fs, order=5):
    """
    Design a low-pass Butterworth filter.

    Args:
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling rate of the signal.
    order (int): The order of the filter.

    Returns:
    b, a (ndarray, ndarray): Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_butter_lowpass_filter(self, data, cutoff, fs, order=5, bool_f = True):
    """
    Apply a low-pass Butterworth filter to a signal.

    Args:
    data (array): The input signal.
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling rate of the signal.
    order (int): The order of the filter.

    Returns:
    y (array): The filtered signal.
    """
    b, a = self._butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    if bool_f:
        # Plotting the data
        plt.figure(figsize=(10, 6))
        plt.plot(data, label='Noisy signal')
        plt.plot(y, label='Filtered signal', linewidth=2)
        plt.xlabel('Time [seconds]')
        plt.ylabel('Voltage [mV]')
        plt.title('Signal before and after Low-pass Filtering')
        plt.legend()
        plt.grid(True)
        plt.show()
    return y
