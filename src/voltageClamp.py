# %%
from neuron import h, gui 
from simulation import Simulation
from simulation_gui import MyWindow
from utils import interpolate_data
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from neuron import h, gui   
import matplotlib as mpl
mpl.rcParams.update({'font.size': 18})




class VoltageClampSimulation(Simulation):
    def __init__(self, data, cell, active_channels, vcParamsConfigPath=None, protocol='Activation', ion_channel_type='Nav'):
        super().__init__(data)
        self.results = {}
        self.cell = cell
        self.active_channels = active_channels
        self.data = data

        # Load the relevant dictionary
        with open(Path(vcParamsConfigPath),"r") as f:
            vcParamsConfig = json.load(f)
        self.vcParams = vcParamsConfig[protocol][ion_channel_type]

        # Setup the recording vectors
        self.recParams = {'voltage' : h.Vector(), 'channel_i': h.Vector(),  'vc_i': h.Vector(), 
                'pas_total' : h.Vector(), 'cap_total' : h.Vector(), 'time' : h.Vector()}
        self.vcParams['voltage_steps'] = np.arange(self.vcParams['V1'], self.vcParams['V2'], self.vcParams['dV'])
        # Build the voltage clamp

        self.pad = 0.5 # Padding for the voltage clamp traces
 
    def _set_cellular_recording(self, cell, ion_channel_name):
        """
        This function sets the cellular recording parameters
        """
        self.recParams['voltage'].record(cell.soma(0.5)._ref_v)
        self.recParams['cap_total'].record(cell.soma(0.5)._ref_i_cap) 

        # (Optional) Set the voltage clamp parameters
        # if ion_channel_type == 'NaV':
        #     self.recParams['channel_i'].record(cell.soma(0.5)._ref_ina)
        # elif ion_channel_type == 'Kv':
        #     self.recParams['channel_i'].record(cell.soma(0.5)._ref_ik) 
        # else:
        self.recParams['channel_i'].record(getattr(cell.soma(0.5), f'_ref_i_{ion_channel_name}'))
    
    def _build_voltage_clamp(self, cell, rs = 0.01):
        """
        This function generates a voltage clamp protocol
        """
        voltageClamp = h.SEClamp(cell.soma(0.5))

        # Set the parameters for the voltage clamp
        voltageClamp.dur1 = self.vcParams['T1']  # ms
        voltageClamp.dur2 = self.vcParams['T2']	 # ms
        voltageClamp.dur3 = self.vcParams['T3']  # ms
        voltageClamp.amp1 = self.vcParams['V0']	 # mV
        voltageClamp.amp2 = self.vcParams['V1']  # mV
        voltageClamp.amp3 = self.vcParams['V0']  # mV
        # TODO: Im
        voltageClamp.rs = rs                      # 2. in some papers   # MOhm

        return voltageClamp

    def _set_voltage_clamp_recording(self, voltageClamp):
        """
        This function sets the voltage clamp recording parameters
        """
        self.recParams['vc_i'].record(voltageClamp._ref_i)
        self.recParams['time'].record(h._ref_t) 
    
    def normalize_current_traces(self, overwrite=False):
        """
        This function normalizes the current traces from peak current to current densities (nA/cm^2) in order for them no to depend on maximum conductance  
        """
        pass

    def simulate_voltage_clamp_protocol(self, channel):
        """
        This function simulates a voltage clamp protocol depending on the initialization with the parameters
        """
        peak_currents = []
        cell = self.cell # Use the prebuilt cell
        self._set_cellular_recording(cell, channel) # Establish recording vectors for the cells
        cell_area = cell.get_area() # Get the area of the cell
        voltageClamp = self._build_voltage_clamp(cell) # Build the voltage clamp
        self._set_voltage_clamp_recording(voltageClamp) # Set the recording vectors for the voltage clamp
        
        # Run the simulation
        self.results[channel] = {}
        self.results[channel]['voltage_steps'] = {}

        for voltage in self.vcParams['voltage_steps']:
            voltageClamp.amp2 = voltage
            
            # Run the simulation
            h.run()

            # Get the peak conductances
            time_trace, curr_density_trace, total_current_trace = self._get_currents(voltageClamp, cell_area)
            # print('Total current trace:', total_current_trace)
            # Store the results
            
            self.results[channel]['voltage_steps'][voltage] = {
                'time': time_trace,
                'curr_density' : curr_density_trace,
                'total_current' : total_current_trace
            }

            peak_currents.append(self.get_peak_current(total_current_trace))
        self.results[channel]['peak_currents'] = peak_currents  # Store the peak currents for the channel

    def _get_currents(self, voltageClamp, area):
        """
        This function calculates the peak conductances from the voltage clamp traces
        """
        start_ipeak_index = int((voltageClamp.dur1+self.pad)/self.data['DT'])
        end_ipeak_index = int((voltageClamp.dur1 + voltageClamp.dur2-self.pad)/self.data['DT'])
        time = np.array(self.recParams['time'])[start_ipeak_index: end_ipeak_index]
        total_current = np.array(self.recParams['vc_i'])[start_ipeak_index: end_ipeak_index] # nA units
        curr_density = total_current/area*10 # TODO: Need to be divided by total capacitance?
        # pdb.set_trace()
        return time, curr_density, total_current
    
    def get_peak_current(self, current_trace):
        """
        This function calculates the maximum conductance for a given channel
        """
        return current_trace[np.argmax(np.abs(current_trace))] # Find the value with the highest magnitude

def plot_voltage_steps(results, channel, ax=None):
    pass

def plot_current_traces(results, channel, current_type = 'total_current', ax=None):
    """W
    This function plots the current traces for a given channel
    Input:
    - results : dict : The results of the simulation of a voltage clamp
    """
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,10))
    voltage_steps = results[channel]['voltage_steps'].keys()
    for voltage in voltage_steps:
        time = results[channel]['voltage_steps'][voltage]['time']
        curr_density = results[channel]['voltage_steps'][voltage][current_type]
        ax.plot(time, curr_density, label=f'Voltage: {voltage} mV')
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'{channel}')
    return ax

def plot_peak_currents(results, ax=None, **kwargs):
    """
    This function plots the peak conductances for a given channel
    Input:
    - results : dict : The results of the simulation of a voltage clamp
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    channels = results.keys()
    for channel in channels:
        peak_currents = results[channel]['peak_currents']
        voltage_steps = results[channel]['voltage_steps_list']
        # Extrapolate the peak conductances
        extrapolate_voltage_steps, extrapolate_peak_currents = interpolate_data(voltage_steps, peak_currents)
        # Plot scatter of original point
        ax.scatter(voltage_steps, peak_currents, **kwargs)
        ax.plot(extrapolate_voltage_steps, extrapolate_peak_currents, label=channel, **kwargs)
    
    ax.grid()
    ax.set_xlabel('Voltage (mV)')
    ax.set_ylabel('Peak currents (nA)')
    ax.set_title('Peak currents for Different Channels')
    return ax


