import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from neuron import h
import efel

# Set non-interactive backend
matplotlib.use('Agg')

# NEURON parallel context
h.load_file("stdrun.hoc")
pc = h.ParallelContext()

# Paths
project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nociceptorCell import Nociceptor

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Parameters
TRP_TYPES = ['trpa1', 'trpv1', 'polymodal', 'control']  # Added control group with no TRP channels
AMPLITUDES = np.arange(0.0, 0.16, 0.01)  # 0.0 to 0.15 nA
N_TRIALS = 5  # Number of repeats per configuration
FIXED_TRP_N = 100 # Standard channel count for this comparison
FIXED_TEMP = 32.0 # Celsius

# Simulation settings
DT = 0.025
STIM_DELAY = 100.0
STIM_DUR = 1000.0
TSTOP = STIM_DELAY + STIM_DUR + 100.0

# Paths
FIG_DIR = project_root / "figures"
DATA_DIR = project_root / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load Cell Config
config_path = project_root / "src" / "configurations" / "cell_config_general.json"
with open(config_path) as f:
    cell_config_data = json.load(f)
# PARAM_NAMES = [
#     'D_OU.somatic', 'gbar_bk.somatic','gbar_sk.somatic','gbar_cal.somatic','gbar_can.somatic',
#     'gbar_cat.somatic','gbar_hd.somatic','gbar_kleak.somatic','gbar_naleak.somatic',
#     'gbar_kv1.somatic','gbar_kv2.somatic','gbar_kv3.somatic','gbar_kv4.somatic',
#     'gbar_kv7.somatic','gbar_nav1p9.somatic','gbar_nav1p8.somatic','gbar_nattxs.somatic'
# ]
# BEST_INDIVIDUAL = [0.013, 0.0012,0.00014,0.00011,0.0001,1e-05,0.0001,1e-05,1e-05,0.0001,0.0002,2e-05,0.003,7e-05,1e-05,0.015,0.0001]

with open("configurations/conductance_configuration.json", 'r') as f:
    param_dict = json.load(f)


# -----------------------------------------------------------------------------
# Job Setup (Group by TRP Type + Amplitude + Trial)
# -----------------------------------------------------------------------------

combos = []
for trp in TRP_TYPES:
    for amp in AMPLITUDES:
        for trial in range(N_TRIALS):
            combos.append({
                'trp': trp,
                'amp': amp,
                'trial': trial
            })

def runid_to_params(run_id):
    return combos[int(run_id)]

# -----------------------------------------------------------------------------
# Worker Function
# -----------------------------------------------------------------------------
def work_fi_simulation(run_id):
    params = runid_to_params(run_id)

    trp_type = params['trp']
    amp = params['amp']
    trial_idx = params['trial']
    
    # 1. Initialize Cell
    cell = Nociceptor()
    cell.build_cell(cell_config_data)
    
    # Add Base Active Conductances
    # Run over key and value
    for param_name, value in param_dict.items():
        base_name, _ = param_name.split('.')
        prefix, mechanism = base_name.split('_')
        cell.add_active_conductances(mechanism, value, prefix=prefix)

    # 2. Insert specific TRP channel(s)
    if trp_type is not 'control':  # Only add channels if not control group
         # Handle polymodal case (both channels) vs single channel case
        if trp_type == 'polymodal':
            channels = ['trpa1', 'trpv1']
        else:
            channels = [trp_type]

        for channel in channels:
            for sec in cell.all:
                sec.insert(channel) 
            for seg in cell.soma:
                mech = getattr(seg, channel)
                mech.N = FIXED_TRP_N
    # else:

    # if trp_type == 'polymodal':
    #     channels = ['trpa1', 'trpv1']
    # else:
    #     channels = [trp_type]

    # for channel in channels:
    #     for sec in cell.all:
    #         sec.insert(channel) 
    #     for seg in cell.soma:
    #         mech = getattr(seg, channel)
    #         mech.N = FIXED_TRP_N

    # 3. Setup Stimulation
    cell.stim.amp = amp
    cell.stim.dur = STIM_DUR
    cell.stim.delay = STIM_DELAY

    # 4. Setup Recording
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(cell.soma(0.5)._ref_v)
    
    # 5. Run Simulation
    h.dt = DT
    h.tstop = TSTOP
    h.finitialize(-65) # Start at RMP
    h.celsius = cell_config_data['TEMPERATURE']  # Fixed temp
    
    h.run()
        
    # 6. Analyze Frequency (Spikes in duration)
    # Simple threshold crossing calculation
    v_arr = np.array(v_vec)
    t_arr = np.array(t_vec)
    
    # Crop to stim window for accurate freq calculation
    start_idx = int(STIM_DELAY / DT)
    end_idx = int((STIM_DELAY + STIM_DUR) / DT)
    
    if start_idx < len(v_arr) and end_idx < len(v_arr):
        v_window = v_arr[start_idx:end_idx]
    else:
        v_window = v_arr # Fallback
        
    # Detect rising edge crossings (> -20 mV)
    spikes = np.where(np.diff((v_window > -20).astype(int)) > 0)[0]
    freq = len(spikes) / (STIM_DUR / 1000.0) # Hz = count / seconds
            
    # Return result dict
    return {
        'trp_type': trp_type,
        'current_nA': amp,
        'trial': trial_idx,
        'frequency_Hz': freq
    }

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Start workers
    pc.runworker()

    if pc.id() == 0:
        total_jobs = len(combos)
        print(f"Starting {total_jobs} F-I curve jobs ({N_TRIALS} trials per config)...")
        print(f"Amplitudes: 0.0 to {AMPLITUDES[-1]:.2f} nA")
        t0 = time.time()

        # Submit jobs
        for i in range(total_jobs):
            pc.submit(work_fi_simulation, i)

        # Collect results
        results = []
        completed = 0
        while pc.working():
            res = pc.pyret()
            results.append(res)
            completed += 1
            if completed % 20 == 0:
                print(f"Job {completed}/{total_jobs} finished.")

        pc.done()
        print(f"Simulations finished in {time.time()-t0:.2f}s")
        
        # Save Results
        df_results = pd.DataFrame(results)
        csv_path = DATA_DIR / 'fig3C_fi_curve_comparison_data.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"Saved data to {csv_path}")

        # -------------------------------------------------------------------------
        # Plotting (Merged F-I Curve)
        # -------------------------------------------------------------------------
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 6))

        # Seaborn Lineplot aggregates trials and shows confidence interval (SEM/CI)
        sns.lineplot(
            data=df_results, 
            x='current_nA', 
            y='frequency_Hz', 
            hue='trp_type', 
            style='trp_type',
            markers=True, 
            dashes=False, 
            linewidth=2, 
            palette='dark',
            errorbar=('ci', 95), 
            err_style='bars', 
        )
        
        plt.xlabel('Current (nA)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'F-I Curve Comparison (Temp: {cell_config_data["TEMPERATURE"]}°C)')
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.legend(title="Model Type")
        plt.tight_layout()
        
        plot_path = FIG_DIR / 'fig3C_fi_curve_comparison_merged.png'
        plt.savefig(plot_path, dpi=300)
        print(f"Saved plot to {plot_path}")
        
    else:
        # Worker process will just wait in runworker()
        pass