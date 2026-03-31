import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as _np
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as _plt
from matplotlib.collections import LineCollection as _LC
from matplotlib.colors import Normalize as _Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_divider
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm 
from utils import sort_underlying_parameters, compute_isi
from process import  get_stacked_currents, get_currentscape



def set_rcoptions(func):
    '''decorator to apply custom matplotlib rc params to function,undo after'''
    import matplotlib

    def wrap(*args, **kwargs):
        """Wrap"""
        options = {'axes.linewidth': 2, }
        with matplotlib.rc_context(rc=options):
            func(*args, **kwargs)
    return wrap

def plot_voltage_trace(t, v, label, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(t, v, label=label)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title('{} Voltage Trace'.format(label))
        return ax



def plot_legend(names, colors):
    """
    Create a legend for the currentscape plot.
    """
    num_items = len(names)
    fig_legend = plt.figure(figsize=(2, num_items * 0.5))
    ax = fig_legend.add_subplot(111)
    ax.axis('off')
    
    legend_patches = [mpatches.Patch(color=colors[i], label=f'{names[i]}') for i in range(num_items)]
    ax.legend(handles=legend_patches, loc='center', frameon=False)
    return fig_legend

def plot_currentscape_w_highlight(data, t_cropped, v_cropped, label, event_time, event_edges, 
                                  area, conv_factor=10, 
                                  all_channel_names=None,
                                  group_map=None, 
                                  highlight_groups=None, 
                                  cmap_name='tab20'):
    """
    Plots currentscape with options to group currents and highlight specific groups.
    
    Parameters:
    - data: Dictionary of  conductances/currents.
    - t_cropped: Time vector.
    - v_cropped: Voltage vector.
    - label: Label for sorting (passed to sort_underlying_parameters).
    - event_time: Time of event to center on (ms).
    - event_edges: Window size around event (ms).
    - area: Cell area for unit conversion.
    - conv_factor: Conversion factor for units.
    - all_channel_names: List of all channel names corresponding to the sorted data keys.
    - group_map: Dictionary mapping individual channel names to group names. 
                 e.g., {'kv1': 'K+', 'kv2': 'K+'}. 
                 If None, no grouping is performed.
    - highlight_groups: List of group names (or channel names if no grouping) to highlight in color.
                        Others will be grey. If None, all are colored.
    """
    
    # 1. Process Data
    current_data, data_dict, data_keys, v_cropped_event, t_cropped_event = sort_underlying_parameters(
        data=data, t_cropped=t_cropped, v_cropped=v_cropped, 
        label=label, event_time=event_time, event_edges=event_edges
    )
    
    # Apply conversion factors
    for key in data_keys:
        data_dict[key] = data_dict[key] * conv_factor * area

    # 2. Group Currents if requested
    if group_map is not None:
        grouped_data_dict = {}
        # Initialize groups with zeros
        unique_groups = []
        for ch_name in all_channel_names:
            grp = group_map.get(ch_name, ch_name) # Default to self if not in map
            if grp not in unique_groups:
                unique_groups.append(grp)
                pass
        
        # Re-stack first to get (n_channels, time)
        raw_currents_names = list(data_dict.keys()) # These are like 'nav1p8'
        print("Raw currents namesnow: ", raw_currents_names)
        # Pop time out of raw_currents_names if present
        # FIXME: A quickfix solution, where the time coming from and why do we need it?
        # FIXME: Remove it
        # if 'time' in raw_currents_names:
        #     raw_currents_names.remove('time')
        raw_stack = get_stacked_currents(data_dict, raw_currents_names) # Shape (n_channels, time)
        
        # Now we aggregate rows based on group_map
        # group_map should map 'LabelName' (from all_channel_names) -> 'GroupName'
        # But we need to link raw_currents_names[i] -> all_channel_names[i] -> GroupName
        
        # Remove time key
        
        if len(raw_currents_names) != len(all_channel_names):
            print(raw_currents_names)
            print("All channel names:")
            print(all_channel_names)
            print(f"Warning: Raw keys {len(raw_currents_names)} != Labels {len(all_channel_names)}")
            # Fallback or strict error
            
        group_to_idx = {}
        grouped_stack_list = []
        final_names = []
        
        temp_grouped_data = {}
        
        for i, raw_key in enumerate(raw_currents_names):
            print(f"Processing {raw_key} -> {all_channel_names[i]}")
            readable_name = all_channel_names[i]
            group_name = group_map.get(readable_name, readable_name)
            
            if group_name not in temp_grouped_data:
                temp_grouped_data[group_name] = np.zeros_like(raw_stack[i])
                final_names.append(group_name)
            
            temp_grouped_data[group_name] += raw_stack[i]
            
        # Re-build matrix
        currents = np.array([temp_grouped_data[name] for name in final_names])
        display_names = final_names
        
    else:
        # No grouping
        raw_currents_names = list(data_dict.keys())
        currents = get_stacked_currents(data_dict, raw_currents_names)
        display_names = all_channel_names


    # 3. Generate Colors
    num_channels = len(display_names)
    cmap_base = plt.get_cmap(cmap_name)
    
    if highlight_groups:
        custom_colors = []         # Create a custom color list
        base_colors = cmap_base.colors if hasattr(cmap_base, 'colors') else [cmap_base(i/num_channels) for i in range(num_channels)]         # Get base colors
        color_idx = 0
        for name in display_names:
            if name in highlight_groups:
                custom_colors.append(base_colors[color_idx % len(base_colors)]) # Assign a color
                color_idx += 1
            else:
                # Assign grey
                custom_colors.append((0.8, 0.8, 0.8, 1.0))
        
        final_colors = custom_colors
    else:
        final_colors = cmap_base.colors[:num_channels] if hasattr(cmap_base, 'colors') else [cmap_base(i/num_channels) for i in range(num_channels)]

    cmap = ListedColormap(final_colors)
    bounds = np.arange(num_channels + 1)
    norm = BoundaryNorm(bounds, cmap.N)

    # 4. Compute Currentscape
    curr_mage, npPD, nnPD = get_currentscape(v_cropped_event, currents)

    # 5. Downsample for plotting
    n = 10
    t_down = t_cropped_event[::n]
    voltage_down = v_cropped_event[::n]
    im0_down = curr_mage[:, ::n]
    npPD_down = npPD[::n]
    
    resy = im0_down.shape[0]

    # 6. Plotting
    swthres = -40
    grid_rows = 5
    
    fig = plt.figure(figsize=(8, 6))
    
    # Voltage (Top)
    ax_top = plt.subplot2grid((grid_rows, 1), (0, 0), rowspan=2)
    ax_top.plot(np.arange(len(voltage_down)), voltage_down, color='black', lw=1.0)
    ax_top.plot(np.arange(len(t_down)), np.ones(len(t_down)) * swthres, ls='dashed', color='black', lw=0.75)
    ax_top.set_ylim(np.min(voltage_down) - 10, np.max(voltage_down) + 10)
    ax_top.set_ylabel('Voltage (mV)')
    ax_top.axis('off') # Hide generic axis
    # Manually adding left spine and label if desired, but code hides them:
    # ax_top.spines['top'].set_visible(False) ...
    # Following original style:
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.xaxis.set_ticks_position('none') 
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Image (Middle)
    ax_img = plt.subplot2grid((grid_rows, 1), (2, 0), rowspan=3, sharex=ax_top)
    ax_img.imshow(im0_down, interpolation='nearest', aspect='auto', cmap=cmap, norm=norm)
    ax_img.plot(resy * 0.5 * np.ones(len(npPD_down)), color='black', lw=2)
    ax_img.set_ylabel('Proportion of Current')
    ax_img.set_xlabel('Time (ms)')
    ax_img.spines['top'].set_visible(False)
    ax_img.spines['right'].set_visible(False)
    ax_img.set_yticks([0, 0.5*resy, resy])
    ax_img.set_yticklabels(['1', '0', '1'])
    
    # Fix X-axis
    ax_top.set_xlim(0, len(t_down) - 1)
    xticks = np.linspace(0, len(t_down) - 1, 5, dtype=int)
    xticklabels = t_down[xticks].round().astype(int)
    ax_img.set_xticks(xticks)
    ax_img.set_xticklabels(xticklabels)

    plt.tight_layout()

    # Legend
    fig_legend = plot_legend(display_names, final_colors)
    fig_legend.show()
    return fig, fig_legend



def plotCurrentscape(voltage, im0, npPD, nnPD):	
    fig = plt.figure(figsize=(6,4))

    # Custom parameters for the plot
    # plt.rcParams['axes.linewidth'] = 0.5
    # plt.rcParams['xtick.major.width'] = 0.5
    swthres=-40  # Threshold for the action potential    

    # Plot the voltage trace
    xmax=len(voltage)
            
    ax=plt.subplot2grid((7,1),(0,0),rowspan=2)	
    t=np.arange(0,len(voltage))
    plt.plot(t, voltage, color='black',lw=1.)
    plt.plot(t,np.ones(len(t))*swthres,ls='dashed',color='black',lw=0.75)
    plt.vlines(1,-50,-20,lw=1)
    plt.ylim(np.min(voltage)-10,np.max(voltage)+10)
    plt.xlim(0,xmax)
    plt.axis('off')         

    # Plot the total inward current in log scale
    ax=plt.subplot2grid((7,1),(2,0),rowspan=1)
    plt.fill_between(np.arange(len((npPD))),(npPD),color='black')
    plt.plot(1.*np.ones(len(nnPD)),color='black', ls=':',lw=1)
    plt.plot(10.*np.ones(len(nnPD)),color='black', ls=':',lw=1)
    plt.plot(100.*np.ones(len(nnPD)),color='black', ls=':',lw=1)
    plt.yscale('log')
    # plt.ylim(0.001,100)
    plt.xlim(0,xmax)
    plt.axis('off') 

    # Currentspace plot
    resy=1000
    elcolormap='Set1'
    ax=plt.subplot2grid((7,1),(3,0),rowspan=3)
    plt.imshow(im0[::1,::1],interpolation='nearest',aspect='auto',cmap=elcolormap)
    plt.ylim(2*resy,0)
    plt.plot(resy*np.ones(len(npPD)),color='black',lw=2)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.xlim(0,xmax)
    plt.clim(0,8)
    plt.axis('off') 

    # Plot total outward current in log scale 
    ax=plt.subplot2grid((7,1),(6,0),rowspan=1)
    plt.fill_between(np.arange(len((nnPD))),(nnPD),color='black')
    plt.plot(1.*np.ones(len(nnPD)),color='black', ls=':',lw=1)
    plt.plot(10.*np.ones(len(nnPD)),color='black', ls=':',lw=1)
    plt.plot(100.*np.ones(len(nnPD)),color='black', ls=':',lw=1)
    plt.yscale('log')
    # plt.ylim(100,0.001)
    plt.gca().invert_yaxis()
    
    plt.xlim(0,xmax)
    plt.axis('off') 
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    return fig

def plot_single_axe_currents(data_dict, data_keys, voltage, time, current_names=None, axes=None,
                             mapping=None, palette=None, alpha=1.0,
                             fontsize=24, fontweight='normal', fontname='Arial',
                             title='', title_fontsize=18, **kwargs):
    """
    Plot voltage (top) and multiple currents on a single lower axis.
    Assumes time, voltage, and currents are already cropped/prepared.
    - mapping: dict mapping data_keys -> human readable names
    - palette: seaborn palette name or list of colors
    Returns: axes
    """

    time = np.asarray(time)
    voltage = np.asarray(voltage)
    n = len(data_keys)

    # build color list
    if palette:
        if isinstance(palette, str):
            colors = sns.color_palette(palette, n)
        else:
            colors = list(palette)[:n]
    else:
        colors = [str(0.5 - 0.3 * i / max(1, n - 1)) for i in range(n)]

    mapping = mapping or {}
    if current_names is None:
        labels = [mapping.get(k, k) for k in data_keys]
    else:
        labels = [mapping.get(k, current_names[i] if i < len(current_names) else k)
                  for i, k in enumerate(data_keys)]

    # create axes if not provided
    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        if title:
            fig.suptitle(title, fontsize=title_fontsize, fontweight=fontweight, fontname=fontname)

    # plot voltage (time in s)
    axes[0].plot(time / 1000.0, voltage, color='black')
    axes[0].set_ylabel('Voltage (mV)', fontsize=fontsize, fontweight=fontweight, fontname=fontname)

    # plot currents on single lower axis
    curr_ax = axes[1]
    for i, key in enumerate(data_keys):
        y = np.asarray(data_dict.get(key, np.zeros_like(time)))
        curr_ax.plot(time / 1000.0, y, color=colors[i], label=labels[i])
        curr_ax.fill_between(time / 1000.0, y, color=colors[i], alpha=alpha * 0.3)

    curr_ax.set_ylabel('I (pA)', fontsize=fontsize, fontweight=fontweight, fontname=fontname)
    curr_ax.legend(loc='lower left', fontsize=max(8, fontsize-6), frameon=False)
    axes[-1].set_xlabel('Time (s)', fontsize=fontsize, fontweight=fontweight, fontname=fontname)

    return axes

def plot_trp_temperature_ramp(t_vec, v_vec, temp_ramp, currents=None,
                             skip_time=60, dt=0.01, figsize=(12,12),
                             title='TRP Activation by Temperature Ramp',
                             spike_times=None, show=True, savepath=None, trp_label='TRPA1',
                             save_csv=None):
    """
    Linear temperature version.
    Plots:
      - Ax1: Voltage vs Time (top axis shows Temperature)
      - Ax2: ISIs vs Time (if spike_times provided, else empty/text)
      - Ax3: Temperature vs Time
    save_csv: None | True | str
      - None: don't save
      - True: save to '<savepath>_data.csv' if savepath given else 'trp_temperature_ramp_data.csv'
      - str: use provided path
    """
    time_array = _np.array(t_vec) / 1000.0
    skip_index = int(skip_time / dt)
    time_array = time_array[skip_index:] - time_array[skip_index]
    v_vec_array = _np.array(v_vec)[skip_index:]
    temp_ramp_cut = _np.asarray(temp_ramp)[skip_index: len(time_array) + skip_index]

    # Normalize currents input to list of (label, array)
    currents_plot = []
    if currents is not None:
        for itm in currents:
            if isinstance(itm, (list, tuple)) and len(itm) == 2:
                lbl, arr = itm
                arr = _np.array(arr)[skip_index: skip_index + len(time_array)]
                currents_plot.append((str(lbl), arr))
            else:
                arr = _np.array(itm)[skip_index: skip_index + len(time_array)]
                currents_plot.append((None, arr))

    # Create 3 subplots now
    fig, (ax1, ax2, ax3) = _plt.subplots(3, 1, figsize=figsize, sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle(title + ' (' + trp_label + ')', fontsize=18)

    # AX1: voltage
    ax1.plot(time_array, v_vec_array, color='blue')
    ax1.set_ylabel('Voltage (mV)')
    
    # Create twin axis for temperature on top of AX1
    ax1_temp = ax1.twiny()
    ax1_temp.set_xlabel('Temperature (°C)')

    # Set limits and ticks for the top temperature axis to match the time-temperature relationship
    t_min, t_max = time_array.min(), time_array.max()
    temp_min, temp_max = temp_ramp_cut.min(), temp_ramp_cut.max()
    
    # 1. Align the limits
    ax1_temp.set_xlim(temp_ramp_cut[0], temp_ramp_cut[-1])
    
    # 2. Ensure ticks are nicely spaced temperatures
    # Let matplotlib decide ticks based on the limits we just set
    
    ax1_temp.spines['top'].set_visible(True)

    # AX2: ISIs
    if spike_times is not None:
        sp = _np.asarray(spike_times, dtype=float)
        sp_sk = sp[sp >= skip_time]
        if len(sp_sk) >= 2:
            isis_sk = compute_isi(sp_sk)  # ms
            if len(isis_sk) > 0:
                x_vals = (sp_sk[:-1] - skip_time) / 1000.0  # s
                y_vals = _np.asarray(isis_sk) / 1000.0
                ax2.scatter(x_vals, y_vals, marker='o', c='black', alpha=0.8, s=30)
                ax2.set_ylabel('ISI (s)')
                ax2.set_yscale('log')
            else:
                ax2.text(0.5, 0.5, 'No ISIs after skip', transform=ax2.transAxes, ha='center')
                ax2.set_ylabel('ISI (s)')
        else:
            ax2.text(0.5, 0.5, 'No spikes after skip', transform=ax2.transAxes, ha='center')
            ax2.set_ylabel('ISI (s)')
    else:
        ax2.text(0.5, 0.5, 'No spikes provided', transform=ax2.transAxes, ha='center')
        ax2.set_ylabel('ISI (s)')

    # AX3: Temperature Ramp
    # Plot the temperature ramp explicitly here
    points = _np.array([time_array[:len(temp_ramp_cut)], temp_ramp_cut]).T.reshape(-1, 1, 2)
    segments = _np.concatenate([points[:-1], points[1:]], axis=1)
    norm = _Normalize(vmin=temp_ramp_cut.min(), vmax=temp_ramp_cut.max())
    lc = _LC(segments, cmap='coolwarm', norm=norm)
    lc.set_array(temp_ramp_cut)
    lc.set_linewidth(3)
    ax3.add_collection(lc)
    ax3.set_ylim(temp_ramp_cut.min() - 1, temp_ramp_cut.max() + 1)
    ax3.set_ylabel('Temp (°C)')
    ax3.autoscale_view() # Ensure the collection is visible
    
    # Add colorbar for temperature axis
    divider = _make_axes_divider(ax3)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    fig.colorbar(lc, cax=cax, label='Temperature (°C)', orientation='vertical')

    # Styling
    for ax in (ax1, ax2, ax3):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(time_array.min(), time_array.max())
    
    ax3.set_xlabel('Time (s)')
    _plt.tight_layout(rect=[0, 0, 1, 0.96])

    # save figure
    if savepath: fig.savefig(savepath, dpi=300, bbox_inches='tight')

    # save csv if requested
    if save_csv:
        if isinstance(save_csv, str):
            csv_path = save_csv
        else:
            if savepath:
                csv_path = savepath.rsplit('.',1)[0] + '_data.csv'
            else:
                csv_path = 'trp_temperature_ramp_data.csv'
        cols = [time_array, v_vec_array, temp_ramp_cut]
        colnames = ['time_s', 'voltage_mV', 'temperature']
        for idx, (lbl, arr) in enumerate(currents_plot):
            name = lbl if lbl is not None else f'current_{idx}'
            cols.append(arr)
            colnames.append(name)
        arr2save = _np.column_stack(cols)
        _np.savetxt(csv_path, arr2save, delimiter=',', header=','.join(colnames), comments='')
        if spike_times is not None and len(sp_sk) >= 2 and len(isis_sk) > 0:
            isi_time = sp_sk[:-1] - skip_time
            isi_df = pd.DataFrame({'isi_time_ms': isi_time, 'isi_ms': isis_sk})
            isi_csv_path = csv_path.rsplit('.',1)[0] + '_featuring_ISIs.csv'
            isi_df.to_csv(isi_csv_path, index=False)
    if show: _plt.show()
    return fig, (ax1, ax2, ax3)

def plot_trp_agonist_ramp(t_vec, v_vec, agonist_ramp, currents=None,
                          skip_time=60, dt=0.01, figsize=(12,12),
                          title='TRP Activation by Agonist Ramp',
                          spike_times=None, show=True, savepath=None, trp_label='TRPA1',
                          save_csv=None,
                          plot_steps=False,           # NEW: if True plot piecewise steps (labels) instead of gradient
                          conc_units='mM',            # unit string for step labels
                          step_label_fmt="{:.0e}"):   # format for concentration labels
    """
    Agonist (log) version: displays agonist axis in log scale and uses log color mapping.
    agonist_ramp: agonist concentration (same length as t_vec), must be >= 0.
    save_csv: same behavior as in plot_trp_temperature_ramp

    New params:
      - plot_steps: if True, draw stepwise (steps-post) agonist trace and annotate each constant segment
      - conc_units: unit string appended to annotated labels
      - step_label_fmt: python format string for concentration values (default scientific)
    """
    eps = 1e-12
    time_array = _np.array(t_vec) / 1000.0
    skip_index = int(skip_time / dt)
    time_array = time_array[skip_index:] - time_array[skip_index]
    v_vec_array = _np.array(v_vec)[skip_index:]
    agonist_cut = _np.asarray(agonist_ramp)[skip_index: len(time_array) + skip_index]
    agonist_cut = _np.maximum(agonist_cut, eps)

    # Normalize currents input to list of (label, array)
    currents_plot = []
    if currents is not None:
        for itm in currents:
            if isinstance(itm, (list, tuple)) and len(itm) == 2:
                lbl, arr = itm
                arr = _np.array(arr)[skip_index: skip_index + len(time_array)]
                currents_plot.append((str(lbl), arr))
            else:
                arr = _np.array(itm)[skip_index: skip_index + len(time_array)]
                currents_plot.append((None, arr))

    fig, (ax1, ax2) = _plt.subplots(2,1, figsize=figsize, sharex=True,
                                             gridspec_kw={'height_ratios':[3,1]})
    fig.suptitle(title + ' (' + trp_label + ')', fontsize=18)

    # AX1: voltage
    ax1.plot(time_array, v_vec_array, color='blue')
    ax1.set_ylabel('Voltage (mV)')
    ax1_temp = ax1.twiny()
    ax1_temp.set_xlabel('Agonist (log scale)')

    # AX2: ISIs if available else agonist-colored line (color uses log10) or step plot
    if spike_times is not None:
        sp = _np.asarray(spike_times, dtype=float)
        sp_sk = sp[sp >= skip_time]
        if len(sp_sk) >= 2:
            isis_sk = compute_isi(sp_sk)  # ms
            if len(isis_sk) > 0:
                x_vals = (sp_sk[:-1] - skip_time) / 1000.0  # s
                y_vals = _np.asarray(isis_sk) / 1000.0
                ax2.scatter(x_vals, y_vals, marker='o', c='black', alpha=0.8, s=30)
                ax2.set_ylabel('ISI (s)')
                ax2.set_xlabel('Time (s)')
                ax2.set_yscale('log')
                ax2.set_xlim(time_array.min(), time_array.max())
            else:
                ax2.text(0.5,0.5, 'No ISIs after skip', transform=ax2.transAxes, ha='center')
                ax2.set_ylabel('ISI (s)')
        else:
            ax2.text(0.5,0.5, 'No spikes after skip', transform=ax2.transAxes, ha='center')
            ax2.set_ylabel('ISI (s)')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    _plt.tight_layout(rect=[0,0,1,0.96])    
    if savepath: fig.savefig(savepath, dpi=300, bbox_inches='tight') # save figure

    # save csv if requested
    if save_csv:
        if isinstance(save_csv, str):
            csv_path = save_csv
        else:
            if savepath:
                csv_path = savepath.rsplit('.',1)[0] + '_agonist_data.csv'
            else:
                csv_path = 'trp_agonist_ramp_data.csv'
        cols = [time_array, v_vec_array, agonist_cut]
        colnames = ['time_s', 'voltage_mV', 'agonist']
        for idx, (lbl, arr) in enumerate(currents_plot):
            name = lbl if lbl is not None else f'current_{idx}'
            cols.append(arr)
            colnames.append(name)
        arr2save = _np.column_stack(cols)
        _np.savetxt(csv_path, arr2save, delimiter=',', header=','.join(colnames), comments='')
        # Save the ISI seperately with time of each ISI 
        if spike_times is not None and len(sp_sk) >= 2 and len(isis_sk) > 0:
            isi_time = sp_sk[:-1] - skip_time
            isi_df = pd.DataFrame({'isi_time_ms': isi_time, 'isi_ms': isis_sk})
            isi_csv_path = csv_path.rsplit('.',1)[0] + '_including_ISIs.csv'
            isi_df.to_csv(isi_csv_path, index=False)

    if show: _plt.show()
    return fig, (ax1, ax2)


def plot_trpv1_temperature_ramp(t_vec, v_vec, temp_ramp, PO_trp, i_trp,
                                skip_time=60, dt=0.01, figsize=(12, 12),
                                title='TRPV1 Activation by Temperature Ramp',
                                savepath=None, show=True):
    """
    Reusable plotting for TRPV1 temperature ramp results.
    t_vec, v_vec, PO_trp, i_trp: vectors/arrays (NEURON Vectors ok)
    temp_ramp: 1D array of temperatures (°C) matching simulation steps
    skip_time: ms to skip at start
    dt: simulation timestep in ms
    savepath: if provided, saves figure to this path
    returns: fig, (ax1, ax2, ax3, ax4)
    """

    # Prepare arrays
    time_array = _np.array(t_vec) / 1000.0  # to seconds
    skip_index = int(skip_time / dt)
    time_array = time_array[skip_index:] - time_array[skip_index]
    v_vec_array = _np.array(v_vec)[skip_index:]
    po_trp_array = _np.array(PO_trp)[skip_index:]
    i_trp_array = _np.array(i_trp)[skip_index:]
    temp_ramp_cut = _np.asarray(temp_ramp)[skip_index: len(time_array) + skip_index]

    # Figure and axes
    fig, (ax1, ax2, ax3, ax4) = _plt.subplots(4, 1, figsize=figsize, sharex=True,
                                             gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    fig.suptitle(title, fontsize=20)

    # Top: voltage
    ax1.plot(time_array, v_vec_array, label='Voltage')
    ax1.set_ylabel('Voltage (mV)')

    # Secondary x-axis showing temperature
    ax1_temp = ax1.twiny()
    ax1_temp.set_xlabel("Temperature (°C)")

    def time_to_temp(t):
        tf = t / (time_array.max() - time_array.min())
        return temp_ramp_cut.min() + tf * (temp_ramp_cut.max() - temp_ramp_cut.min())

    ax1_temp.set_xlim(time_to_temp(ax1.get_xlim()[0]), time_to_temp(ax1.get_xlim()[1]))
    ax1_temp.spines['top'].set_visible(True)

    # Temperature ramp colored line
    points = _np.array([time_array[:len(temp_ramp_cut)], temp_ramp_cut]).T.reshape(-1, 1, 2)
    segments = _np.concatenate([points[:-1], points[1:]], axis=1)
    norm = _Normalize(vmin=temp_ramp_cut.min(), vmax=temp_ramp_cut.max())
    lc = _LC(segments, cmap='coolwarm', norm=norm)
    lc.set_array(temp_ramp_cut)
    lc.set_linewidth(3)
    ax2.add_collection(lc)
    ax2.set_ylim(temp_ramp_cut.min(), temp_ramp_cut.max())
    ax2.set_ylabel('Temp (°C)')

    # Colorbar beside ax2
    divider = _make_axes_divider(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(lc, cax=cax, label='Temperature (°C)', orientation='vertical')

    # PO_trp and i_trp plots
    ax3.plot(time_array, po_trp_array, color='green')
    ax3.set_ylabel('Open probability')

    ax4.plot(time_array, i_trp_array, color='purple')
    ax4.set_ylabel('TRPV1 current (nA)')
    ax4.set_xlabel('Time (s)')

    # Clean up aesthetics
    for ax in (ax1, ax2, ax3, ax4):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(time_array.min(), time_array.max())

    _plt.tight_layout(rect=[0, 0, 1, 0.96])

    if savepath:
        fig.savefig(savepath, format='jpeg', dpi=300)
    if show:
        _plt.show()

    return fig, (ax1, ax2, ax3, ax4)


def plot_spike_analysis(t_vec, v_vec, spike_times, isis, tstop, dt,
                        figsize=(8, 9), bins=40, show=True):
    """
    Plot voltage (top), firing-rate histogram (middle), ISI scatter (bottom).
    t_vec: time vector (ms)
    v_vec: voltage vector (mV)
    spike_times: array-like spike times (ms)
    isis: inter-spike intervals (ms)
    tstop: simulation stop time (ms)
    dt: timestep (ms)
    """
    fig, axes = _plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=False)

    # 1) Top: Voltage trace
    axes[0].plot(_np.array(t_vec) / 1000.0, _np.array(v_vec), color='blue')
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_title('Voltage Trace')
    axes[0].set_xlim([0, tstop / 1000.0])

    # 2) Middle: Firing rate histogram
    axes[1].hist(spike_times, bins=bins, color='orange')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Count')
    axes[1].set_xlim([0, tstop])
    axes[1].set_title('Firing Rate')

    # 3) Bottom: ISI scatter
    isis_s = _np.array(isis) / 1000.0
    axes[2].scatter(spike_times[:-1], isis_s, marker='o', c='black', alpha=0.7)
    axes[2].set_ylabel("ISI (s)")
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_yscale("log")
    axes[2].set_title("Inter-Spike Intervals")
    axes[2].set_xlim([0, tstop])

    _plt.tight_layout()
    if show:
        _plt.show()
    return fig, axes

