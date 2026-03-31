from datetime import datetime
from warnings import catch_warnings
import numpy as np
from neuron import h
import pylab as p
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec as SubGridSpec
import mplcursors

# Append to path
import os
import time
from pathlib import Path
import json

# Setup logging
import logging
log = logging.getLogger(__name__)


# parent_folder = Path(__file__).parent.parents[1]
# file_path = parent_folder / Path(r"src/configuration/variables_fitted_nseg.json")
# with open(Path(file_path),"r") as f:
#     data = json.load(f)


class Simulation():
    def __init__(self, data, cell=None, set_recording_vectors=True):
        # Define variables
        self.id = 0
        self.rheobase = None
        self.ap_height = None   
        
        # Create dictionary for saving parameters
        self.saved_parameters = {} 

        # Extract data fields
        self.record_location = data['RECORDING_LOCATION']
        self.threshold = data['THRESHOLD']
        self.celsius = data['TEMPERATURE']
        self.v_init = data['V_INIT']
        self.tstop = data['TSTOP']
        self.dt = data['DT']
        self.duration = data['DUR']
        self.delay = data['DELAY']
        self.min_current = data['MIN_CURRENT']
        self.max_current = data['MAX_CURRENT']
        self.rheobase_step = data['RHEOBASE_STEP']
        self.current_step = data['STEP']
        
        # Initialize variables
        self.init_simulation()
        self.rec_channels = {}
        self.rec_gates = {}
        
        if cell:
            self.cell = cell
            self._setup_AP_recorder()

            if set_recording_vectors: 
                # FIXME: Make it more organized and less redudant
                self.recording_params = cell.recording_params
                self.recording_params_array = cell.recording_params_array
                self.recording_array_lengths = cell.recording_array_lengths
                self.recording_gating_params = cell.recording_gating_params


        # Set default parents folder
        self.parent_folder = Path(__file__).parent.parents[1]
        
    def __del__(self):
        self.cell.remove_from_neuron()

    
    def _setup_AP_recorder(self):
        # TODO: Move to cell
        # %% Initialize APCount
        # Initialize APCounter
        # ap_vector = h.Vector()
        for sec in h.allsec():
            self.apc = h.APCount(sec(self.record_location))
            self.apc.thresh = self.threshold
        # self.apc = h.APCount(self.cell.soma(self.record_location))
        # Threshold for spike detection in mV
        # self.apc_times = self.apc.record(ap_vector)

    def init_simulation(self):
        """Initialize and run a simulation.
        :param celsius:
        :param v_init:
        :param tstop: Duration of the simulation.
        """
        # Initialize the simulation environment
        # h.load_file("stdrun.hoc")
        
        # Setup parameters
        h.v_init = self.v_init
        h.tstop = self.tstop
        h.celsius = self.celsius
        h.dt = self.dt
    
    def run_simulation(self, vinit, tstop):
        """
        Custom simulation protocol for running a particular step
        """
        h.finitialize(vinit) # Initialize the membrane potential and recording vectors
        if h.cvode.active():
            h.cvode.active()
            print("CVode is active")
        else:
            h.fcurrent()
        h.frecord_init() # Reinitialize recording vectors
        h.continuerun(h.t+tstop)

    def set_parameters(self, seg, param_dict):
        """
        Set NEURON Simulator variable stored in param_dict keys to corresponding values
        """
        for k,v in param_dict.items():
            setattr(seg, '%s' % k, v)

    def _add_recorders(self, segment, labels, rec=None, chan = None, array_l=None):
        # Always have a time vector recorded
        # if rec is None:
        #     rec = {'t': h.Vector()}
        #     rec['t'].record(h._ref_t) 
        if rec is None:
            rec = {}

        if array_l:
            for label, label_values in labels.items():
                for value in label_values:
                    label_name = label + '_' + value
                    rec[label_name] = {}
                    for i in range(array_l):
                        rec[label_name]["%s"%(i)] = h.Vector()
                        # label = labels[label][i]
                        exestr = "rec['%s']['%s'].record(segment.%s._ref_%s[%s])" %( label_name, i, label, value,i)
                        try:
                            exec(exestr)
                        except ValueError:
                            print("Error in recording {}".format(value))
        else:
            # if chan:
            for label, label_values in labels.items():
                for value in label_values:
                    label_name = label + '_' + value
                    rec[label_name] = h.Vector()

                    exestr = "rec['%s'].record(segment.%s._ref_%s)" %( label_name, label, value)
                    try: 
                        exec(exestr)
                    except ValueError:
                        print("Error in recording {}".format(value))
            # else: 
            #     for label in labels:
            #         rec[label] = h.Vector()
            #         exestr = "rec['%s'].record(segment._ref_%s)" %( label, label)
            #         exec(exestr)
        return rec
    
    def set_recording_vectors(self, segment, channel_mechanisms=None):
        """
        """
        # Set the initial mechanisms
        if not channel_mechanisms:
            channel_mechanisms = self.recording_params
        self.rec_dict = self._add_recorders(segment, labels=channel_mechanisms)

        # Add the mechanisms for the  channels
        # FIXME: Make it more robust than an if
        if self.recording_array_lengths:
            for array_l, recording_param_arr in zip(self.recording_array_lengths, self.recording_params_array):
                self.rec_channels = self._add_recorders(segment, labels=recording_param_arr, array_l=array_l, rec=self.rec_channels) 
        
        # Set recording vectors for gating variables
        if self.recording_gating_params:
            self.rec_gates = self._add_recorders(segment, labels=self.recording_gating_params)
        
        return self.rec_dict, self.rec_channels, self.rec_gates
    


        # for ch_name, channel_mechanism in channel_mechanisms.items():
        #     self.rec_dict = self.makeRecorders(segment, rec=self.rec_dict, labels = {'i_{}'.format(ch_name): '_ref_i_{}'.format(channel_mechanism)})
        # nav_states = 4
        # kv_states = 2
        # # Specify makeRecovers with a list of labels 
        # self.rec_dict = self.makeRecorders(segment.pas, labels = {'v': '_ref_i'})
        # self.rec_dict = self.makeRecorders(segment, rec=self.rec_dict, labels = {'ik': '_ref_ik', 'ina': '_ref_ina'})
        # # self.rec_dict = self.makeRecorders(segment.nav1p7, rec=self.rec_dict, labels = {'nav1p7_m': '_ref_m', 'nav1p7_h': '_ref_h'})
        # self.rec_dict = self.makeRecorders(segment.nattxs, rec=self.rec_dict, labels = {'nav1p7_m': '_ref_m', 'nav1p7_h': '_ref_h'})
        # self.rec_dict = self.makeRecorders(segment.nav1p9mkv_org, rec=self.rec_dict, labels = {'nav1p9_n': '_ref_Nast'}, array_l=nav_states)
        # self.rec_dict = self.makeRecorders(segment.kv7mrkv, rec=self.rec_dict, labels = {'kv7_n': '_ref_Kst'}, array_l=kv_states)
        # # self.rec_dict = self.makeRecorders(segment.nav1p9mkv, rec=self.rec_dict, labels = {'nav1p9_evNa': '_ref_next_evNa', 'nav1p9_RNA': '_ref_nextRNa'})
        # # self.rec_dict = self.makeRecorders(segment.kv7mrkv, rec=self.rec_dict, labels = {'kv7_evNa': '_ref_next_evK', 'kv7_RNA': '_ref_nextRK'})
        # return self.rec_dict  
    
    def makeRecorders(self, segment, labels, rec=None, array_l=0):
        '''
        Make recorders for a segment
        Note to access a variable in a segment, use the following syntax:
            "rv.record(sec(0.5)._ref_%s_%s)" %( label, chan )

        For example:
            rv.record(sec(0.5)._ref_v)
            rv.record(sec(0.5).nav1p9mkv._ref_Nast)

        Here's a sample example on how to record with cVode
            cvode = h.CVode()
            tvec = h.Vector()                                                            
            Vrec = h. Vector() 
            cvode.record(cell.soma(0.5).nav1p9mkv.Nast,Vrec,tvec) 
            
        :param segment: The segment to record from
        :param labels: A dictionary of labels and the corresponding variable to record
        :param rec: A dictionary of recorders to add to
        :return: A dictionary of recorders
        '''
        
        if rec is None:
            rec = {'t': h.Vector()}
            rec['t'].record(h._ref_t) 

            
        # elif not(chan): 
        #     rec = {}
        #     for label in labels:
        #         rv = h.Vector()
        #         exestr = "rv.record(sec(0.5)._ref_%s_%s)" %( label, chan )
        #         exec(exestr)
        #         rec[label] = rv
        # cvode = h.CVode()
        if array_l > 0:
            for label in labels:
                rec[label] = {}
                for i in range(array_l):
                    rec[label][i] = h.Vector()
                    rec[label][i].record(getattr(segment, labels[label])[i])
        else:
            for k,v in labels.items():
                rec[k] = h.Vector()
                rec[k].record(getattr(segment, v))
        return rec
    
    def makeVclamp(self, cell, dur= 100, amp=-40, v_start=-100, v_step=5, v_stop=-10, channel_type='Na'):
        """
        TODO: Implement
        TODO: Add channel blockage to each condition
        """
        # while (h.t<h.tstop): # runs a single trace, calculates peak current
        #     dens = f3cl.i/segment.area()*100.0-segment.i_cap # clamping current in mA/cm2, for each dt
        #     t_vec.append(h.t)       # code for store the current
        #     v_vec_t.append(segment.v)  # trace to be plotted
        #     i_vec.append(dens)      # trace to be plotted
            
        #     if ((h.t>=540)and(h.t<=542)):     # evaluate the peak (I know it is there)
        #         if(abs(dens)>abs(peak_curr)):
        #             peak_curr = dens        
        #             t_peak = h.t
                    
        #     h.fadvance()

        # updates the vectors at the end of the run
        # # %% Create voltage clamp experiment
        # voltageClamp = h.VClamp(0.5, sec=cell.soma)
        voltageClamp = h.SEClamp(cell.soma(0.5))

        # %% Voltage clamp experiment
        cell.stim.amp = 0.0           # nA
        voltageClamp.dur1 = 400	      # ms
        voltageClamp.amp1 = -100	  # mV
        voltageClamp.dur2 = dur       # ms
        voltageClamp.amp2 = amp       # mV
        voltageClamp.dur3 = 400       # ms
        voltageClamp.amp3 = -100       # mV
        voltageClamp.rs = 0.01 # 2. in some papers   # MOhm
        # voltageClamp._ref_i = h.Vector()
        vcParams = {'voltage' : h.Vector(), 'I_peak': h.Vector(),  'vc_i': h.Vector(), 
                    'pas_total' : h.Vector(), 'cap_total' : h.Vector(), 'time' : h.Vector()}
        vcParams['voltage'].record(cell.soma(0.5)._ref_v)
        if channel_type == 'Na':
            vcParams['I_peak'].record(cell.soma(0.5)._ref_ina)
        elif channel_type == 'K':
            vcParams['I_peak'].record(cell.soma(0.5)._ref_ik)
        vcParams['voltage'].record(cell.soma(0.5)._ref_v)
        vcParams['cap_total'].record(cell.soma(0.5)._ref_i_cap)
        vcParams['vc_i'].record(voltageClamp._ref_i)
        vcParams['time'].record(h._ref_t)   
        vcParams['voltage_steps'] = np.arange(v_start, v_stop, v_step)  

        return voltageClamp, vcParams

    def get_peak_conductance(self, cell, voltageClamp, vcParams, channel_type='Na'):
        """
        TODO:
        """
        
        max_I = []
        Ipeak_currents = []
        area = cell.get_area()

        # Define a figure with a 3x2 grid and an additional axis spanning 2 columns
        fig = plt.figure(figsize=(12.5, 8.5))
        gs = fig.add_gridspec(3, 2)
        ax = [[None, None], [None, None], None]
        ax[0][0] = fig.add_subplot(gs[0, 0])
        ax[0][1] = fig.add_subplot(gs[0, 1])
        ax[1][0] = fig.add_subplot(gs[1, 0])
        ax[1][1] = fig.add_subplot(gs[1, 1])
        ax[2] = fig.add_subplot(gs[2, :])

        for voltage in vcParams['voltage_steps']:
            voltageClamp.amp2 = voltage
            
            # Run the simulation
            h.run()

            # Plot voltage clamp results
            ax[0][0].plot(vcParams['time'], vcParams['voltage'], color='black', label='Voltage (mV)')
            ax[1][0].plot(vcParams['time'], vcParams['I_peak']*area*10)
            ax[0][1].plot(vcParams['time'], vcParams['cap_total'], color='blue')
            ax[1][1].plot(vcParams['time'], vcParams['vc_i'], color='green')


            # Save the variable
            time = np.array(vcParams['time']) # TODO: Improve redundancy
            start_ipeak_index = int(voltageClamp.dur1/h.dt)
            # start_ipeak_index = np.where(time == start_ipeak)[0][0]
            end_ipeak_index = int((voltageClamp.dur1 + voltageClamp.dur2)/h.dt)
            # end_ipeak_index = np.where(time == end_ipeak)[0][0]
            I_peak = np.array(vcParams['I_peak'])[start_ipeak_index: end_ipeak_index]
            I_peak = I_peak*area*10
            max_I.append(max(abs(I_peak)))
            # if channel_type == 'Na':
            #     I_peak = np.array(vcParams['I_peak'])[start_ipeak: end_ipeak]
            #     I_peak = I_peak*area*10
            #     max_I.append(min(I_peak))
            # elif channel_type == 'K':
            #     I_peak = np.array(vcParams['I_peak'])[start_ipeak: end_ipeak]
            #     I_peak = I_peak*area*10
            #     max_I.append(max(I_peak))
            Ipeak_currents.append(vcParams['I_peak']*area*10)
        
        # Set labels and titles for the plots
        ax[0][0].set_xlabel('Time (ms)')
        ax[0][0].set_ylabel('Voltage (mV)')
        ax[0][0].set_title('Voltage Clamp mV')
        ax[1][0].set_xlabel('Time (ms)')
        ax[1][0].set_ylabel('Current (pA)')
        ax[1][0].set_title('I_peak Current')
        ax[0][1].set_xlabel('Time (ms)')
        ax[0][1].set_ylabel('Capcita (mA/cm^2)')
        ax[0][1].set_title('Capacitive Current')
        ax[1][1].set_xlabel('Time (ms)')
        ax[1][1].set_ylabel('$I_{vc}$ (mA/cm^2)')
        ax[1][1].set_title('VC Current')

        # Set xlimits for all axis
        right_lim = voltageClamp.dur1 - 10
        left_lim = voltageClamp.dur1 + voltageClamp.dur2 + 10
        ax[0][0].set_xlim(right_lim, left_lim)
        ax[1][0].set_xlim(right_lim, left_lim) 
        ax[0][1].set_xlim(right_lim, left_lim)
        ax[1][1].set_xlim(right_lim, left_lim)

        # Plot on the big axis at the bottom
        ax[2].scatter(vcParams['voltage_steps'], max_I, color='red', label='Max Ina (mA/cm^2)')
        ax[2].set_xlabel('Voltage (mV)')
        ax[2].set_ylabel('Max Ina (pA)')
        plt.suptitle('Voltage Clamp Results', fontsize=26)
        plt.tight_layout()
        plt.show()


        return max_I, Ipeak_currents, time
    
    def get_normalized_data(self, data):
        min_val = min(data)
        max_val = max(data)
        # data = [(x - min_val)/(max_val - min_val) for x in data]
        data  =  [(x - min_val)/(max_val - min_val) for x in data]
        return data
   
    def get_voltage_trace(self, cell):
    
        # Setup parameters
        h.run()
        
        # Do it without and with flatten
        ttrace = np.array(cell.t_vec)
        vtrace = np.array(cell.soma_v_vec)
        
        # Action potential calculation
        ap_n = self.apc.n 

        return ttrace, vtrace, ap_n
    
    
    def plot_voltage_trace(self, ttrace, vtrace, 
                            new_current, nNa='', nNk='', gk_max = '', gna_max = '',
                            zoom_region=None, save_fig=True, *args, **kwargs):
        # Plotting
        fig, ax = plt.subplots(1,1)
        ax.plot(ttrace, vtrace, color='black',*args, **kwargs)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title('Current Injection: ' + str(new_current) + ' nA')
        ax.set_ylim(-80, 30)
        
        # Display zoom in region
        if zoom_region is not None:
            x_start, x_end, y_start, y_end = zoom_region
            axins = ax.inset_axes([0.7, 0.6, 0.25, 0.35])
            axins.plot(ttrace, vtrace, *args, **kwargs)
            axins.set_xlim(x_start, x_end)
            axins.set_ylim(y_start, y_end)
            ax.indicate_inset_zoom(axins)
            mplcursors.cursor(axins)

        # Text for the box
        textstr = 'Model Parameters:\ngk_max = {} (S/cm^2), gna_max={}(S/cm^2)'.format(gk_max, gna_max)
        # Properties of the box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # Place a text box in the bottom of the plot
        ax.text(0.5, -0.2, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='center', bbox=props)
        
        # Save the plot
        if save_fig:
            self.save_figure_plot(fig, 'voltage_trace_' + str(new_current) + 'nA')
        plt.show()
    
    def get_voltage_trace_batch(self, cell, current_injection_protocol=None , 
                                duration = None, plot_flag = False):
        
        # Parse the None arguements
        if current_injection_protocol is None:
            current_injection_protocol = np.arange(self.min_current, self.max_current, self.current_step) # Injection current in nA
        if duration is None:
            duration = self.duration

        
        # Initialize lists
        time_data = []
        voltage_data = []
        ap_data = []
        
        for new_current in current_injection_protocol:
            cell.stim.amp = new_current # FIXME: Verify this works
            self.get_voltage_trace(cell)
            
            # Do it without and with flatten
            ttrace = np.array(cell.t_vec)
            vtrace = np.array(cell.soma_v_vec)
            
            time_data.append(ttrace)
            voltage_data.append(vtrace)
            # Action potential calculation
            ap_n = self.apc.n / (duration/1000)
            ap_data.append(ap_n)
        
        if plot_flag:
            self.plot_voltage_trace_batch(time_data, voltage_data, ap_data, current_injection_protocol)
        
        return time_data, voltage_data, ap_data, current_injection_protocol
            
    def plot_voltage_trace_batch(self, time_data, voltage_data, ap_data, current_injection_protocol, save_fig=True, *args, **kwargs):
        # Get the voltage trace
        # Setup graph: Plotting everything on a single graph
        inject_amount = len(current_injection_protocol)
        fig, ax = plt.subplots()
        for i in range(inject_amount):
            ax.plot(time_data[i], voltage_data[i], label=str(current_injection_protocol[i]*1e3) + ' pA, Spike Number = {}'.format(ap_data[i]))
            # Format graph
        plt.xlabel('time (ms)')
        # plt.ylabel('mV')
        plt.yticks([])
        ax.axhline(0, color='r', linestyle='dashed')
        ax.legend(prop={'size': 2})
        frame = plt.gca()
        frame.spines['top'].set_visible(False)
        frame.spines['right'].set_visible(False)
        frame.spines['left'].set_visible(False)

        
        if save_fig:
            self.save_figure_plot(fig, 'voltage_trace_batch')    
 


    def brute_parameter_search(simulation_func, parameter_ranges, num_points):
        """
        Perform a brute parameter search space for running a simulation in a neuron simulator.
        
        Parameters:
            simulation_func (function): The function that runs the simulation using the parameters.
            parameter_ranges (dict): A dictionary of parameter ranges for each parameter.
                                    The keys represent the parameter names, and the values are tuples (min_value, max_value).
            num_points (int): The number of points to sample within each parameter range.
            
        Returns:
            best_parameters (dict): The best set of parameters found during the search.
            best_result (float): The result obtained using the best set of parameters.
        """
        # Generate parameter combinations
        parameter_combinations = []
        for param_name, (min_value, max_value) in parameter_ranges.items():
            values = np.linspace(min_value, max_value, num_points)
            parameter_combinations.append((param_name, values))
        
        # Initialize best result and parameters
        best_result = float('-inf')
        best_parameters = {}
        
        # Perform the parameter search
        for combination in np.meshgrid(*[values for _, values in parameter_combinations]):
            parameters = {param_name: value for (param_name, _), value in zip(parameter_combinations, combination)}
            result = simulation_func(parameters)
            
            if result > best_result:
                best_result = result
                best_parameters = parameters
        
        return best_parameters, best_result

    def get_if(self, cell, dur = None, current_arr = None,  plot_f = True):
        """
        Extract the frequency output curve of the cell
        """
        # Parse the steps
        if dur is None:
            dur = self.duration
        if current_arr is None:
            current_arr = np.arange(self.min_current, self.max_current, self.current_step)

        segment = cell.soma(0.5)
        aps = []
        
        # Instill action potential counter
        ap = h.APCount(segment)

        # Loop over injection
        for inj in current_arr:
            cell.stim.amp = inj
            h.run()

            #Number of action potentials - Divided by sampling time
            ap_n = ap.n / (dur/1000)
            aps.append(ap_n)
        
        if plot_f:
            self.plot_if(current_arr, aps, cell)        

    def plot_if(self, current_vector, freq_vector, cell):
        """
        Description: Plots the IF curves of 
        """
        fig = plt.figure(figsize=(12, 12))        
        plt.plot(current_vector, freq_vector ,marker='o',color='black', linewidth=2)
        # ax1.plot(current_vector, freq_onset_vector, '--', color=color_vec[1][0], label = currlabel + " onset rate")
        plt.xlabel("Current [nA]")
        plt.ylabel("Frequency [Hz]")
        plt.title("I/F")
        lg = plt.legend()
        lg.get_frame().set_linewidth(0.5)
        # TODO: Fix the save figure
        self.save_figure_plot(fig, 'channel_conductances') 
        # plt.savefig('cell_data/figures/' + cell.label + 'IF_Curve.svg', format = 'svg', bbox_inches = 'tight', dpi = 300)
        # plt.savefig('cell_data/figures/' + cell.label + 'IF_Curve.png', format = 'png', bbox_inches = 'tight', dpi = 300)
        plt.show()
        
    
    def get_channel_properties(self, cell, chan = 'nav1p9', vars_labels = {"Nart[20]"}, plot_figure = True):
        """
        Description: Extracts the channel properties from the neuron
        """




        # Record variables of interest
        for vars_label in vars_labels:
            rec = self.makeRecorders(cell.soma(0.5), {}.format(vars_label))
        


        if plot_figure:
            # Create figure
            fig = plt.figure(figsize=(20,10), dpi=200)
            fig.subplots_adjust(0.15, 0.092, 0.99, 0.88)

            grid = GridSpec(2, 2, wspace=0.4, hspace=0.2)
            ax0 = fig.add_subplot(grid[0, 0])
            #ax0.set_xlabel('V (mV)')
            ax0.set_ylabel('Steady state of activation')
            ax1 = fig.add_subplot(grid[0, 1])
            #ax1.set_xlabel('V (mV)')
            ax1.set_ylabel('Steady state of inact.')
            ax2 = fig.add_subplot(grid[1, 0])
            ax2.set_xlabel('V (mV)')
            ax2.set_ylabel('Time constant of act. (ms)')
            ax3 = fig.add_subplot(grid[1, 1])
            ax3.set_xlabel('V (mV)')
            ax3.set_ylabel('Time constant of inact. (ms)')



        # Get inactivation and activation curve of the channel (m_inf, h_inf as a function of time)

        # Get the time constant as a function of voltage
        # Get the channel conductance as a function of time


        
    def plot_channel_properties(self, func, variables, title_head='Channel Properties'):
        # Set up the figure and subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns

        # A - Sigmoidal curves
        voltage = np.linspace(-150, 100, 300)
        # Calculate using the lambda functions

        m_inf = func['m_inf'](voltage, variables)
        h_inf = func['h_inf'](voltage, variables)
        axs[0, 0].plot(voltage, h_inf, 'k--', label='h (Sim.)')
        axs[0, 0].plot(voltage, m_inf, 'k-', label='m (Sim.)')
        if 'm_exp' in func:
            axs[0, 0].scatter(func['m_voltage_arr'], func['m_exp'], label='m (Exp.)', color='blue')
        if 'h_exp' in func:
            axs[0, 0].scatter(func['h_voltage_arr'], func['h_exp'], marker='^', label='h (Exp.)', color='blue')
        axs[0, 0].set_xlabel('Membrane Potential, Vₘ (mV)')
        axs[0, 0].set_ylabel('Steady-state Parameter')
        axs[0, 0].legend()
        axs[0,0].set_xlim(-80, 20) # View only the relevant area

        # B - Activation time constant 
        tau_m = func['tau_m'](voltage, variables)
        axs[0, 1].plot(voltage, tau_m, 'k-', label='τₙ (Sim.)')
        axs[0, 1].set_xlabel('Membrane Potential, Vₘ (mV)')
        axs[0, 1].set_ylabel('Time Constant of Activation (ms)')
        axs[0, 1].legend()

        # C - Inactivation time constant
        tau_h = func['tau_h'](voltage, variables)
        axs[0, 2].plot(voltage, tau_h, 'k--', label='$τ_{h}$ (Sim.)')
        axs[0, 2].set_xlabel('Membrane Potential, Vₘ (mV)')
        axs[0, 2].set_ylabel('Time Constants of Inactivation (ms)')
        axs[0, 2].legend()

        # D - Time series data with multiple traces
        for i, current in enumerate(func['currents']):
            axs[1, 0].plot(func['time'], current, label = str(func['voltage_steps'][i]) + ' nA')
        axs[1, 0].set_xlabel('Time (ms)')
        axs[1, 0].set_ylabel('I (pA)')
        axs[1, 0].legend()

        # E - Monotonically increasing function
        axs[1, 1].plot(func['voltage_steps'], func['I_peak'], 'k-', label='Peak $I_k$ (Sim.)')
        axs[1, 1].set_xlabel('Membrane Potential, Vₘ (mV)')
        axs[1, 1].set_ylabel('Peak I (pA)')
        axs[1, 1].legend()

        # Plot the alpha and beta functions
        axs[1, 2].plot(voltage, func['alpha_m'](voltage, variables), 'k-', label='αₙ (Sim.)')
        axs[1, 2].plot(voltage, func['beta_m'](voltage, variables), 'k--', label='βₙ (Sim.)')
        axs[1, 2].set_xlabel('Membrane Potential, Vₘ (mV)')
        axs[1, 2].set_ylabel('Rate Constant (ms⁻¹)')
        axs[1, 2].legend()


        # Adjust the layout
        plt.suptitle(title_head, fontsize=28)
        plt.tight_layout()

        # Show the plot
        plt.show()
    
    def _get_ap_amplitude(self, cell, ap_time):
        """
        Get amplitude of action potential
        Achieves so getting action potential 100 ms within first threshold cross
        Inputs:
         - cell - Neuron model with all characteristics
         - ap_time - The time of the action potential
        """
        v_np_array = np.array(cell.soma_v_vec)
        # t_np_array = np.array(cell.t_vec)
        ap_index_start = int(ap_time / self.dt)      #
        ap_index_end = ap_index_start + int(50/self.dt) # + 50 ms range
        ap_height = max(v_np_array[ap_index_start:ap_index_end])
        # print('ap_time: ',ap_time)
        # print('ap_time: ',t_np_array[ap_index_start])
        print('The AP height is {} mV'.format(ap_height))
        return ap_height

    def get_reversal_potential(self, cell):
        ttrace, vtrace, ap_n = self.get_voltage_trace(cell)
        self.plot_voltage_trace(ttrace, vtrace, zoom_region=(700, 900, -58, -52), new_current=0)
        # Get the reversal potentials
        RMP = np.mean(vtrace)
        print('The resting membrane potential is {} mV'.format(RMP))
        return RMP

    def get_input_resistance(self, cell, delay=None, dur=None, plot_flag=True, save_to_excel = False):
        """
        Description: Extracts the input resistance from the neuron 

        The protocol from the experiment was:
        (RIN) was measured in voltage-clamp mode using negative 10- to 40-mV pulses from a holding level of –80 mV. 
        Only cells with a resting potential (VR) negative to –60 mV were included into this study RIN of 1.7 GΩ
        
        For the original mode the Rin was assumed to be around 1.7G, According to recorded cell which were 1.7 ± 0.3 GΩ
        """
        # Set the delay and duration
        if delay is None:
            delay = self.delay
        if dur is None:
            dur = self.duration
        
        # Define variables
        segment = cell.soma(0.5)
        current_arr = [-0.1,-0.02,0.02]
        rec = self.makeRecorders(segment, {'v': '_ref_v'})
        ap = h.APCount(segment)
        ap.thresh = -20
        spks = h.Vector()
        ap.record(spks)
        I = []
        V = []


        if plot_flag:
            p.figure()
            p.subplot(1,2,1)

        for k,i in enumerate(np.arange(current_arr[0],current_arr[1],current_arr[2])):     
            spks.clear()
            ap.n = 0
            cell.stim.amp = i
            h.run()
            spike_times = np.array(list(spks)) # Redudant and fix
            if len(np.intersect1d(np.nonzero(spike_times>delay)[0], np.nonzero(spike_times<delay+dur)[0])) == 0:
                # Recording of time calculated in ms   
                t = np.array(rec['t'])
                # Recording of voltage in mV
                v = np.array(rec['v'])
                # Extract steady state of the voltage
                idx = np.intersect1d(np.nonzero(t > delay+0.75*dur)[0], np.nonzero(t < delay+dur)[0])
                # Insert current
                I.append(i)
                # Calculate the mean of the steady state state, and substract the resting voltage pot
                V.append(np.mean(v[idx]) + 70) 
            else:
                print('The neuron emitted spikes at I = %g pA' % (cell.stim.amp*1e3))
            
            # Convert to different units:
            t_new_units = 1e-3*t  # What units is this?


            if plot_flag:
                p.plot(1e-3*t,v)
            
        #? Covert to microvolt, why ar we doing this? to help with the fit?        
        V = np.array(V)*1e-3
        # Convert current to pA units
        I = np.array(I)*1e-9

        #? Verify the polyfit function
        poly = np.polyfit(I,V,1)
        if plot_flag:
            # Format the plot and plot the results
            ymin,ymax = p.ylim()
            p.plot([1e-3*(delay+0.75*dur),1e-3*(delay+0.75*dur)],[ymin,ymax],'r--')
            p.plot([1e-3*(delay+dur),1e-3*(delay+dur)],[ymin,ymax],'r--')
            p.xlabel('t (s)')
            p.ylabel('V (mV)')
            p.box(True)
            p.grid(False)
            p.subplot(1,2,2)
            
            # Plots the current injected
            x = np.linspace(I[0],I[-1],100)
            y = np.polyval(poly,x)
            p.plot(1e12*x,1e3*y,'k--')
            p.plot(1e12*I,1e3*V,'bo')
            p.xlabel('I (pA)')
            p.ylabel('V (mV)')
            # Save the figures
            # plt.savefig('cell_data/figures/' + cell.label + '_IV_Rin_Protocol.svg', format = 'svg', bbox_inches = 'tight', dpi = 1200)
            # plt.savefig('cell_data/figures/' + cell.label + '_IV_Rin_Protocol.png', format = 'png', bbox_inches = 'tight', dpi = 1200)
            p.show()       
        
        #Convert to MegaOhm
        Rin = poly[0]*1e-6
        
        # Save the data
        # np.save('cell_data/figures/IV_data', [V, I])
        print('The cell input resistance is ' + str(Rin) + ' MOhm')

        # Save the data to excel


        # Return the data
        return Rin
    
    def get_phase_plane_trace(self, cell):
        ais_list = [5, 60]
        # Make stim object       
        # Plot the phase plane of the graphs
        fig1 = p.figure(1)      
        fig1_1 = fig1.add_subplot(221)
        fig1_2 = fig1.add_subplot(223)
        colors=["orangered","darkred","gold"]
        for index, item in enumerate(ais_list):
            cell.spacer.L = item
            self.rheobase_protocol(cell)
            cell.stim.amp = self.rheobase
            print("Iinj =", self.rheobase, "nA")
            h.run()    
            time = np.array(cell.t_vec)
            vtrace=np.array(cell.soma_v_vec).flatten()
            # sliced_vtrace = vtrace[round(740/DT):round(760/DT)]
            # sliced_time = time[round(740/DT):round(760/DT)]
            
            # vtraceAIS=np.array(cell.AIS_v_vec).flatten()
            dv= self.extract_dv_dt(vtrace)
            # dvAIS= self.extract_dv_dt(vtraceAIS) 
            # Plot the data 
            fig1_1.plot(time, vtrace, color=colors[index])
            # fig1_1.plot(time, vtraceAIS, color =colors[index], linestyle="dashed")     
            fig1_2.plot(vtrace[: len(dv)], dv, color=colors[index])
            # fig1_2.plot(vtraceAIS[: len(dvAIS)], dvAIS,linestyle="dashed", color=colors[index])   
    
        # Edit the files
        fsize = 10
        fig1_1.set_xlabel("Time [ms]", fontsize=fsize)
        fig1_1.set_ylabel("Voltage [mV]", fontsize=fsize)
        # fig1_1.set_xlim([700,800])
        fig1_2.set_xlabel("Voltage [mV]", fontsize=fsize)    
        fig1_2.set_ylabel("dV/dt [V/s]", fontsize=fsize)
        # Format the graph
        #format the plot    
        p.figure(1)
        # fig1_1.set_title("Iinj = 0.4 nA (red), 0.8 nA (red), 1.3 nA (orange) \n Soma (solid) and AIS (dashed)")
        fig1_1.set_title("Bright red {} AIS dark red {} AIS".format(5,60))
        p.show()
    
    def get_rheobase(self, cell, min_current = None, 
                     max_current = None, 
                     rheobase_step = None, plot_flag = False):
        # Parse the data
        if min_current is None:
            min_current = self.min_current
        if max_current is None:
            max_current = self.max_current
        if rheobase_step is None:
            rheobase_step = self.rheobase_step
        # Get the rheobase
        
        apc = h.APCount(cell.soma(0.5))
        apc.thresh = 0
        current_lst = np.arange(min_current, max_current, rheobase_step)
        for new_current in current_lst:
            cell.stim.amp = new_current
            h.run()
            if apc.n > 0:
                self.ap_height = self._get_ap_amplitude(cell, apc.time)
                self.rheobase = new_current
                if plot_flag:
                    # Plot the results
                    plt.plot(cell.t_vec, cell.soma_v_vec, label=str(new_current) + ', Rheobase = {}'.format(new_current))
                    plt.suptitle('Spike Graph', fontsize=14, fontweight='bold')
                    plt.axvline(x = apc.time, color = 'r')
                    plt.axhline(y = self.ap_height, color = 'b', linestyle = '-')
                    # plt.text(0.1, 2.8, "The number of action potentials is {}".format(apc.n))
                    plt.xlabel('time (ms)')
                    plt.ylabel('mV')
                    plt.xlim(600,700)
                    plt.legend()
                    plt.show()
                    # Get ap height:
                log.info('The rheobase is {} nA'.format(self.rheobase))
                return self.rheobase
        
    def extract_dv_dt(self, vtrace):
        '''2-point first order finite difference to estimate dV/dt '''
        dt = self.dt
        dv = []
        for i in range(1, len(vtrace)-2): 
            dv.append((vtrace[i+1]-vtrace[i-1])/(2*dt))
            # dv.append((vtrace[i+1]-vtrace[i-1])/(dt))
        return dv

    def create_output_filename(self, prefix='', extension='.h5'):
        """
        Create output filename for saving dat
        """
        filename = prefix
        if prefix != '' and prefix[-1] != '_':
            filename = filename + '_'
        now = time.localtime(time.time())
        filename = filename + '%d%02d%02d-%02d%02d%02d' % \
            (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
        if extension[0] != '.':
            extension = '.' + extension
        suffix = ''
        k = 0
        while os.path.exists(filename + suffix + extension):
            k = k+1
            suffix = '_%d' % k
        return filename + suffix + extension
    
    def save_figure_plot(self, fig, name, parent_folder=None, *args, **kwargs):
        '''
        Save the figure as a png and svg file in the figures folder
        '''
        if parent_folder is None:
            parent_folder = self.parent_folder
        
        # Create the parent folder with the current date 
        current_datetime = datetime.now() # Get the current date and time
        # Format the datetime as a date timestamp (YYYY-MM-DD)
        date_timestamp = current_datetime.strftime('%Y-%m-%d')
        # Format the datetime as an hour-minute timestamp (HH:MM)
        hour_minute_timestamp = current_datetime.strftime('%H%M%S')
        
        # Create folder path
        folder_path = parent_folder / 'figures' / str(date_timestamp) 
        os.makedirs(folder_path, exist_ok=True)  # Create the directory with the timestamp
        # Save the figure
        file_path_svg = folder_path / Path(r"{}_{}.svg".format(name,hour_minute_timestamp))
        file_path_png = folder_path / Path(r"{}_{}.png".format(name,hour_minute_timestamp))
        fig.savefig(file_path_svg, format = 'svg', bbox_inches = 'tight', dpi = 300)
        fig.savefig(file_path_png, format = 'png', bbox_inches = 'tight', dpi = 300)
        

    def save_simulation(self, cell, title="Simulation", t_vec=None, v_vec=None,**kwargs) -> None:
        """
        TODO: Implement usage of kwargs for additional arguemnts
        Save the simulation data into a H5 file
        
        Parameters:
            filename (str): The name of the file to save the data into.
            cell (object): The cell object that was used for the simulation.
            **kwargs: The parameters that were used for the simulation.

        Returns:
            None
        """

        # Save cellular parameters of the simulation
        self.saved_parameters['Specific membrane Rm'] = cell.soma.g_pas
        self.saved_parameters['Specific membrane Cm'] = cell.soma.cm
        self.saved_parameters['Specific membrane Ra'] = cell.soma.Ra
        self.saved_parameters['Soma length'] = cell.soma.L
        self.saved_parameters['Soma diam'] = cell.soma.L
        self.saved_parameters['NaV 1.9 gbar'] = cell.soma.gnabar_nav1p9mkv
        self.saved_parameters['NaV 1.9 NNa'] = cell.soma.NNa_nav1p9mkv
        # self.saved_parameters['NaV 1.8 gbar'] = cell.soma.gnabar_nav1p8mkv
        # self.saved_parameters['NaV 1.8 NNa'] = cell.soma.NNa_nav1p8mkv
        self.saved_parameters['Kv gbar'] = cell.soma.gkbar_kmdrgmrkv
        self.saved_parameters['Kv NK'] = cell.soma.NK_kmdrgmrkv
        if t_vec is None or v_vec is None:
            t_vec = cell.t_vec
            v_vec = cell.soma_v_vec
        else:
            t_vec = t_vec
            v_vec = v_vec
        self.saved_parameters['t'] = list(t_vec)
        self.saved_parameters['v'] = list(v_vec)
        # Save the parameters in the dictionary
        for key, value in kwargs.items():
            self.saved_parameters[key] = value
        # self.saved_parameters['ina'] = np.array(cell.soma_ina_vec).astype('float64')
        # self.saved_parameters['ik'] = np.array(cell.soma_ik_vec).astype('float64')

        # Create filename 
        filename = self.create_output_filename(prefix="Simulation", extension='.json')
        filepath = Path(__file__).parent.parent / 'data/results'/ filename

        # TODO: Improve by using the h5py library
        with open(filepath, 'w') as fp:
            json.dump(self.saved_parameters, fp)

    def load_simulation_data(self, file_path):
        """
        FIXME: Implement simulation loading
        Load simulation data from a JSON file.
        Parameters:
            file_path (str): The path to the JSON file containing the simulation data.

        Returns:
            dict: A dictionary containing the loaded simulation data.
        """
        # Check if file exists
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"No file found at {file_path}")

        # Load data from the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Convert lists to numpy arrays if necessary
        if 't' in data:
            data['t'] = np.array(data['t'])
        if 'v' in data:
            data['v'] = np.array(data['v'])

        return data

    def record_specific_channel_current(channel_mechanisms):
        """Record the current through a specific channel mechanism"""
        channel_currents = {}
        for ch_name, channel_mechanism in channel_mechanisms.items():
            channel_currents[ch_name] = h.Vector()
            eval("channel_currents[ch_name].record(cell.soma(0.5)._ref_i_{})".format(channel_mechanism)) 
        return channel_currents   

    def run_rheobase_protocol(self, cell, N_runs):
        rheobase_values = []
        for _ in range(N_runs):
            rheobase = self.get_rheobase(cell)  # Assumes get_rheobase is a function you've defined elsewhere
            rheobase_values.append(rheobase)
        return np.mean(rheobase_values)  # Return the average rheobase value from all runs

    def explore_brute_parameter_space(self, cell, bounds, param_names, N_runs=3, n_samples= 11, plot_f=False):
        """
        Function for exploring brute parameter space as a function of two channels
        For example ;Nav 1.7, Nav1.8. The goal would be to test

        params: 
        - cell: neuron cell model from 
        - bounds: list of tuples, each containing the lower and upper bounds for the parameter space
        - param_names; list of the parameter names
        - N_runs : number of runs 
        """
        X_arr = np.linspace(bounds[0][0], bounds[0][1], n_samples)
        Y_arr = np.linspace(bounds[1][0], bounds[1][1], n_samples)
        rheobase_matrix = np.zeros((len(X_arr), len(Y_arr)))
        
        # Iterate over the parameter space
        for i, x_gbar in enumerate(X_arr):
            for j, y_gbar in enumerate(Y_arr):

                seg = self.cell.soma(0.5)  # Generalize to all sections if needed
                setattr(seg, '%s' % param_names[0], x_gbar)
                setattr(seg, '%s' % param_names[1], y_gbar)
                
                # Submit tasks to ParallelContext
                rheobase_matrix[i, j] = self.run_rheobase_protocol(cell, N_runs)
                # self.pc.submit(self.run_rheobase_protocol, N_runs)
        
        # Gather results
        # while (self.pc.working()):
        #     # FIXME: Make sure the processes finish on time
        #     rheobase_matrix[i, j] = self.pc.pyret()
        # my_id = int(self.pc.id())  # Get the ID of the current process
        # # nhost = int(self.pc.nhost())  # Get the total number of processes
        # for i in range(len(X_arr)):
        #     for j in range(len(Y_arr)):
        #         if (i * len(Y_arr) + j) % nhost == my_id:
        #             rheobase_matrix[i, j] = self.pc.pyret()

        # Reduce data across all hosts
        # self.pc.allreduce(rheobase_matrix, 1)  # 1 for SUM operation
        if plot_f: # Plot only from the master process
            import matplotlib.pyplot as plt
            plt.title('Rheobase matrix')
            plt.imshow(rheobase_matrix, cmap='hot', interpolation='nearest')
            plt.show()

        return rheobase_matrix
