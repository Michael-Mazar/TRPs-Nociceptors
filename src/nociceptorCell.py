from matplotlib import pyplot as plt
from neuron import h
import numpy as np
import csv
import json
from pathlib import Path
import os
import random as rnd

class Nociceptor():
    """A ball & stick neuron model describing """
    def __init__(self):
        self.all = h.allsec()
        self.recording_params = {}
        self.recording_gating_params = {}
        self.recording_array_lengths = None
        self.recording_params_array = None

    def __repr__(self):
        return 'Nociceptor'
    
    def load_mechanisms(self):
        "TODO: Fix to make it work in code to mechanism can be loaded seperately"
        """ Load the mechanisms. """
        if os.path.exists("nrnmech.dll"):
            self.write_log(f"Loading nrnmech.dll")
            h.nrn_load_dll("nrnmech.dll")
        elif os.path.exists("x86_64"):
            self.write_log(f"Loading x86_64/.libs/libnrnmech.so")
            h.nrn_load_dll("x86_64/.libs/libnrnmech.so")
        else:
            self.write_log("No compiled mechanisms found. If you use custom mechanisms you need to run nrnivmodl")
    
    def build_cell(self, data):
        self.create_sections()
        self.define_geometry(data)
        self.set_passive_biophysics(data)
        self.add_stim_object(data)
        # self.add_current_stim(ramp_flag=f_ramp)

    def create_sections(self):
        """ Create morphological sections """
        self.soma = h.Section(name='soma', cell=self)
        
    def define_geometry(self, data):
        '''Define Length, Diamter and        Number of Segment per Section'''
        # Soma
        self.soma.L = data['SOMA_L']
        self.soma.diam = data['SOMA_DIAM']
        self.soma.nseg = data['SOMA_NSEG']  
        
    def set_passive_biophysics(self, data):
        '''Set cell biophyisics including passive and active properties '''
        # Set passive membrane biophysics
        for sec in self.all:
            sec.Ra = data['R_A']
            sec.cm = data['C_M']
 
    def add_passive_leak(self, data):
        # Set leaky channels for the model:
        for sec in self.all:
            sec.insert('pas')
            sec.g_pas = data['GPAS']  # in this version its .pas.g instead of g_pas
            sec.e_pas = data['E_PAS']
    
                
    def add_stim_object(self, data):
        """Attach a current Clamp to a cell."""
        self.stim = h.IClamp(self.soma(data['RECORDING_LOCATION']))
        self.stim.delay = data['DELAY']
        self.stim.dur = data['DUR']
        self.stim.amp = data['INIT_CURRENT'] # nA

    def add_active_conductances(self, channel, gmax, gates=None, channel_arr= None, channel_states=None, prefix=None):
        """Add active ion channel conductances to the neuron model."""
        for sec in self.all:
            sec.insert(channel)
            if prefix is not None:
                setattr(sec, f"{prefix}_{channel}", gmax)
            else:
                setattr(sec, f"gbar_{channel}", gmax)
        
        # Setup the recording parameters
        self.recording_params[channel] = ["i"]
        if gates is not None:
            self.recording_gating_params[channel] = gates
        
        # FIXME: Recording param_array and length as well
        if channel_arr is not None:
            self.recording_params_array = channel_arr
            self.recording_array_lengths= channel_states

    def add_current_stim(self, current, delay, dur):
        """Attach a current Clamp to a cell.
        :param cell: Cell object to attach the current clamp.
        :param delay: Onset of the injected current.
        :param dur: Duration of the stimulus.
        :param amp: Magnitude of the current.
        :param loc: Location on the dendrite where the stimulus is placed.
        """
        self.stim.amp = current
        self.stim.delay = delay
        self.stim.dur = dur
    
    def add_noise_stim(self, current, noise_std, delay, dur, tstop, DT):
        """Attach a current Clamp to a cell."""
        # Set the defaults to 0
        self.stim.delay=0
        self.stim.dur=1e9
        self.stim.amp =0 # nA
        
        # Set 
        noise_pulse = np.zeros(int(tstop/DT)) # ms
        tvec = h.Vector(np.linspace(delay, tstop, int(tstop/DT)))
        stim_timepoints = [0,int(dur/DT)]
        duration = int(dur/DT)
        delayed= int(delay/DT)
        noise_pulse[stim_timepoints[0]+delayed:stim_timepoints[0]+delayed+duration] = np.random.normal(current, noise_std, int(np.diff(stim_timepoints)[0]))
        noise_current_vector = h.Vector(noise_pulse)
        # noise_current_vector.from_python(noise_pulse)
        noise_current_vector.play(self.stim._ref_amp, DT)

        # Check played vector is the same as the one stored
        assert np.all(noise_current_vector.as_numpy() == noise_pulse)
        
        # Store the vector
        self.stim_vec = noise_current_vector
        self.amp = current

    def add_action_potential_recording(self, segment):
        """Attach a current Clamp to a cell."""
        apc = h.APCount(segment)
        apc.thresh = -20
        self.apc = apc
        self.spks = h.Vector()
        apc.record(self.spks)
    
    def add_ramp_stim(self, current, delay,
                             dur, dt, tstop, min_ramp=0):
        """
        Attach to all the simulation processes
        """
        # add prestep
        self.stim.delay=0
        self.stim.dur=1e9
        self.stim.amp =0 # nA
        
        ramp_delay_ms = delay
        ramp_delay = int(ramp_delay_ms/dt)
        min_ramp = min_ramp# nA
        max_ramp = current # nA
        ramp_len = dur # ms
        ramp_arr = np.zeros(int(tstop/dt)) # ms
        ramp_range = [0,int(ramp_len/dt)] # The duration of the ramp
        ramp_arr[ramp_range[0]+ramp_delay:ramp_range[1]+ ramp_delay] = np.linspace(min_ramp,
                                                                            max_ramp,
                                                                            int(np.diff(ramp_range)[0]))
        # ramp_arr[ramp_range[1]+ ramp_delay:2*(ramp_range[1])+ramp_delay] = np.linspace(max_ramp,
                                                                            # min_ramp,
                                                                            # int(np.diff(ramp_range)[0]))
        # prestep
        prestep_delay_ms = 30
        prestep_delay = prestep_delay_ms/dt
        min_prestep = 0.25*min_ramp
        max_prestep = 0.25*max_ramp
    
        stim_vec = h.Vector(ramp_arr)
        stim_vec.play(self.stim._ref_amp, dt)
        self.stim_vec = stim_vec
        self.stim.amp = current
    
    def add_partitioned_ramp_stim(self, currents, ramps_lens, delay,
                                  dt, tstop):
        # add prestep
        self.stim.delay=0
        self.stim.dur=1e9
        self.stim.amp =0 # nA

        ramp_arr = np.zeros(int(tstop/dt)) # ms
        # ramp_range = [0,int(ramps_lens[0])] # The duration of the ramp
        ramp_delay = int(delay/dt)
        # Loop over the remaining elements and inject into the ramp
        for i in range(1, len(currents)):
            ramp_range = [ramp_delay, ramp_delay+int(ramps_lens[i-1]/dt)]
            ramp_arr[ramp_range[0]:ramp_range[1]] = np.linspace(currents[i-1], currents[i], int(np.diff(ramp_range)[0]))
            ramp_delay = ramp_range[1]

        stim_vec = h.Vector(ramp_arr)
        stim_vec.play(self.stim._ref_amp, dt)
        self.stim_vec = stim_vec
        self.stim.amp = currents[-1]


    # Voltage clamp the cell
    def set_recording(self):
        """Set soma, axon initial segment, and time recording vectors on the cell.
        :param cell: Cell to record from.
        :return: the soma, dendrite, and time vectors as a tuple.
        """
        self.soma_v_vec = h.Vector()  # Membrane potential vector at soma  # Membrane potential vector at dendrite
        self.t_vec = h.Vector()  # Time stamp vector
        self.soma_v_vec.record(self.soma(0.5)._ref_v)
        self.t_vec.record(h._ref_t)
    
    def get_area(self, f_point_neuron=True):
        """Get the surface area of the cell.
        :param cell: Cell object to get the surface area.
        :return: The surface area of the cell.
        Note: Te formula for 2pi*r*h yields the same results
        """
        totalarea = 0
        for sec in self.soma.wholetree():
                for seg in sec:
                    totalarea += h.area(0.5, sec=sec)  
        return totalarea
    
    def get_total_capacitance(self):
        """Get the total capacitance of the cell.
        :param cell: Cell object to get the capacitance.
        :return: The capacitance of the cell.
        """
        # %% Calculate theoretical : 
        Cm_goal = 30 # pF
        # A = 2*np.pi*cell.soma.L*cell.soma.diam/2 + 2*np.pi*(cell.soma.diam/2)**2 # 3000 um^2
        A = self.get_area() # um squared
        A = A*1e-8 #  um^2 to cm^2
        cm = 1*1000000 # uF/cm^2 from pF/cm^2
        Cm_theoretical = cm*A  # pF/um^2 / um^2 = pF
        print('Cm = ', Cm_theoretical, 'pF', 'whereas Cm goal is = ', Cm_goal, 'pF')
        return Cm_theoretical, Cm_goal
    
    def get_reversal_potential(self):
        h.run()
        vtrace = np.array(self.soma_v_vec)
        trace_time = np.array(self.t_vec)   
        
        
        # ap_n = self.apc.n 
        # RMP = np.mean(vtrace)
        # Check for AP and if not present calculate RMP, otherwise return NaN
        if self.apc.n > 0:
            print('The neuron emitted spikes, so RMP cannot be calculated accurately.')
            return None, vtrace, trace_time
        # Otherwise calculate RMP as the median of the voltage trace (to avoid outliers from noise)
        plt.figure()
        plt.plot(np.array(self.t_vec), vtrace)
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        RMP = np.median(vtrace)
        print('The resting membrane potential is {} mV'.format(RMP))
        return RMP, vtrace, trace_time
    
    def get_input_resistance(self, delay=None, dur=None, plot_flag=True, save_to_excel = False):
        """
        Description: Extracts the input resistance from the neuron 

        The protocol from the experiment was:
        (RIN) was measured in voltage-clamp mode using negative 10- to 40-mV pulses from a holding level of –80 mV. 
        Only cells with a resting potential (VR) negative to –60 mV were included into this study RIN of 1.7 GΩ
        
        For the original mode the Rin was assumed to be around 1.7G, According to recorded cell which were 1.7 ± 0.3 GΩ
        """
        # Define variables
        segment = self.soma(0.5)
        current_arr = [-0.1,-0.02,0.02]
        rec = self.makeRecorders(segment, {'v': '_ref_v'})
        ap = h.APCount(segment)
        ap.thresh = -20
        spks = h.Vector()
        ap.record(spks)
        I = []
        V = []

        # ensure t/v are always available (will be updated each run)
        t = None
        v = None

        if plot_flag:
            plt.figure()
            plt.subplot(1,2,1)

        for k,i in enumerate(np.arange(current_arr[0],current_arr[1],current_arr[2])):     
            spks.clear()
            ap.n = 0
            self.stim.amp = i
            h.run()

            # read recorded time and voltage every iteration so variables are always set
            t = np.array(rec['t'])
            v = np.array(rec['v'])

            spike_times = np.array(list(spks)) # Redudant and fix
            if len(np.intersect1d(np.nonzero(spike_times>delay)[0], np.nonzero(spike_times<delay+dur)[0])) == 0:
                # Recording of time calculated in ms   
                # Extract steady state of the voltage
                idx = np.intersect1d(np.nonzero(t > delay+0.75*dur)[0], np.nonzero(t < delay+dur)[0])
                # Insert current
                I.append(i)
                # Calculate the mean of the steady state state, and substract the resting voltage pot
                V.append(np.mean(v[idx]) + 70) 
            else:
                print('The neuron emitted spikes at I = %g pA' % (self.stim.amp*1e3))
            
            # Convert to different units for plotting (if needed)
            if plot_flag:
                plt.plot(1e-3*t,v)
            
        # If no valid IV points were collected, avoid polyfit failure
        if len(I) == 0 or len(V) == 0:
            print('No non-spiking IV points collected; cannot compute Rin.')
            return np.nan

        #? Covert to microvolt, why ar we doing this? to help with the fit?        
        V = np.array(V)*1e-3
        # Convert current to pA units
        I = np.array(I)*1e-9

        #? Verify the polyfit function
        poly = np.polyfit(I,V,1)
        if plot_flag:
            # Format the plot and plot the results
            ymin,ymax = plt.ylim()
            plt.plot([1e-3*(delay+0.75*dur),1e-3*(delay+0.75*dur)],[ymin,ymax],'r--')
            plt.plot([1e-3*(delay+dur),1e-3*(delay+dur)],[ymin,ymax],'r--')
            plt.xlabel('t (s)')
            plt.ylabel('V (mV)')
            plt.box(True)
            plt.grid(False)
            plt.subplot(1,2,2)
            
            # Plots the current injected
            x = np.linspace(I[0],I[-1],100)
            y = np.polyval(poly,x)
            plt.plot(1e12*x,1e3*y,'k--')
            plt.plot(1e12*I,1e3*V,'bo')
            plt.xlabel('I (pA)')
            plt.ylabel('V (mV)')
            # Save the figures
            # plt.savefig('cell_data/figures/' + cell.label + '_IV_Rin_Protocol.svg', format = 'svg', bbox_inches = 'tight', dpi = 1200)
            # plt.savefig('cell_data/figures/' + cell.label + '_IV_Rin_Protocol.png', format = 'png', bbox_inches = 'tight', dpi = 1200)
            plt.show()       
        #Convert to MegaOhm
        Rin = poly[0]*1e-6
 
        # Save the data
        # np.save('cell_data/figures/IV_data', [V, I])
        print('The cell input resistance is ' + str(Rin) + ' MOhm')
        return Rin
    
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
    def remove_from_neuron(self):
        for k, v in self.__dict__.items():
            del v
