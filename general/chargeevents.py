import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    ## Charge Events
    """)
    return


@app.cell
def _():
    import marimo as mo
    import sys 
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.colors import LogNorm
    import pandas as pd
    import copy
    from prettytable import PrettyTable
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    from collections import defaultdict
    colors = sns.color_palette('colorblind')


    # apply sexy Latex matplotlib style
    plt.style.use('/home/pgrams/latex-cm.mplstyle')


    # Add the location of the data utilities code to the system path
    # Can also just add it to PYTHONPATH but make sure you know where it is pointing
    # or you could end up using the wrong code.

    # The root directory for the utility code is up 2 directories from the notebook
    abs_repo_path = os.path.abspath('/home/pgrams/tpc_data/software/PGramsRawData/')
    # Insert the path to the front of sys.path if it is not already there
    if not abs_repo_path in sys.path:
        sys.path.insert(0, abs_repo_path)
    return curve_fit, defaultdict, find_peaks, mo, np, pd, plt


@app.cell
def _():
    import raw_data_ana.get_data as get_raw_data
    import raw_data_ana.data_checks as data_checks
    import raw_data_ana.plotting as plot
    import raw_data_ana.charge_utils as qutils
    import raw_data_ana.light_utils as lutils
    return (get_raw_data,)


@app.cell
def _(curve_fit, decoder_bindings, find_peaks, np, pd, plt, readout_df):
    def plot_charge_channels(adc_words, event, num_channel, timesize, charge_range=[1950, 2150]):
        xdown, xup = -int(timesize/2), timesize

        plt.figure(figsize=(18,6))
        plt.imshow(adc_words, cmap=plt.cm.RdBu_r, extent=[xdown, xup, 0, num_channel], vmin=charge_range[0], vmax=charge_range[1], origin='lower')
        plt.plot([0, 0], [0, num_channel], linestyle='--', color='gray')
        plt.plot([xup / 2, xup / 2], [0, num_channel], linestyle='--', color='gray')
        clb=plt.colorbar()
        clb.set_label('Charge  [ADC]')
        plt.xticks(np.arange(-150, 300, 50))
        plt.yticks(np.arange(0, num_channel+1, 16))

        plt.title("Event " + str(event))
        plt.xlabel("[$\\mu$s]")
        plt.ylabel("Channel")
        plt.xlim(xdown,xup)
        plt.show()

    def plot_difference(evt, channel):
        shift = 4
        approx_baseline = readout_df['light_adc_words'][evt][channel, 0]
        diff = readout_df['light_adc_words'][evt][channel, shift:].astype(float) - readout_df['light_adc_words'][evt][channel, :-shift].astype(float)
        diff[diff < 0] = 0
        plt.plot(np.concatenate((np.ones(shift)*approx_baseline, diff+approx_baseline)), linestyle='--')

    # 2. Define the fitting function (e.g., a Gaussian)
    def gaussian(x, mu, sigma):
        return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)

    def gaussian_exp(x, amplitude, mean, stddev, tau):
        exp = np.exp(-(x - mean) / tau)
        exp[x <= mean] = 0
        guass = amplitude * (np.exp(-((x - mean) / (2 * stddev))**2)  + exp)
        guass[x > mean] /= 2
        return guass 

    def fit_histogram(h, cb, frac=0.3):
        # 3. Use scipy.optimize.curve_fit to fit the histogram
        bc = (cb[:-1] + cb[1:]) / 2
        mask = h > frac * np.max(h)
        popt, pcov = curve_fit(gaussian, bc[mask], h[mask], p0=[max(h), 2000, 200])
        # popt, pcov = curve_fit(gaussian_exp, bc[mask], h[mask], p0=[max(h), 2050, 25,1])
        return bc[mask], popt

    def plot_trigger_rate(df, fem_num, bin_range, framesize=255, axis=None):
        '''
        Reconstruct trigger rate

        Time between trigger is,
        Δs = trig_sample[i] - trig_sample[i-1]
        Δf = trig_frame[i] - trig_frame[i-1]
        framesize = trigger framesize in 2MHz ticks, likely 255 = 128μs for pGRAMS
        ΔT = Δf * framesize + Δs
        '''
        f1 = df['event_frame_number'].str[fem_num][:-1].values.astype(float)
        f2 = df['event_frame_number'].str[fem_num][1:].values.astype(float)
        # f1 = df['trigger_frame_number'].str[fem_num][:-1].values.astype(float)
        # f2 = df['trigger_frame_number'].str[fem_num][1:].values.astype(float)

        s1 = df['trigger_sample'].str[fem_num][:-1].values.astype(float)
        s2 = df['trigger_sample'].str[fem_num][1:].values.astype(float)

        trig_time_diff = (f2 - f1) * framesize + (s2 - s1)
        trig_time_diff[trig_time_diff == 0] = 1e-10

        if axis is None:
            c,_,_ = plt.hist(1. / (trig_time_diff * 5.e-7), bins=50, range=bin_range)
            plt.xticks(np.arange(bin_range[0],bin_range[1]+1, 50))
            plt.xlabel('Trigger Rate [Hz]  (from event $\\Delta$t)')
        else:
            c,_,_ = axis.hist(1. / (trig_time_diff * 5.e-7), bins=50, range=bin_range, alpha=0.8, density=True)
            axis.set_xticks(np.arange(bin_range[0],bin_range[1]+1, 50))
            axis.set_xlabel('Trigger Rate [Hz]  (from event $\\Delta$t)')

        print("Number of events in histogram:", np.sum(c))
        print(c)

        # plt.show()
        return 1. / (trig_time_diff * 5.e-7)    

    def concat_row_arrays(row):
      return np.concatenate(row)

    def get_event_data(files, key_list=None, get_hits_only=False):
        file_list = [files] if type(files) is not list else files

        readout_data = []
        evt_cnt = 0
        for fname in file_list:
            proc = decoder_bindings.ProcessEvents(light_slot=16)
            proc.open_file(fname)
            while proc.get_event():
                try:
                    tmp = proc.get_event_dict()
                    if get_hits_only:
                        if not are_hits(readout_df=tmp, event=None):
                            continue
                    if key_list is not None: tmp = {key: tmp[key] for key in key_list}
                    readout_data.append(tmp)
                    evt_cnt += 1
                except:
                    continue
                if evt_cnt == 5000:
                    break


        # Data frame with 1 event per row
        readout_df = pd.DataFrame(readout_data)

        return readout_df, proc



    # modifying are_hits to also find negative pulses and the large dropout events 
    def are_hits(readout_df, event):
        """
        Use for event selection. Check if there are any large-ish charge hits in the event in any of the 32 channels.
        """
        charge_channels = readout_df['charge_adc_words'] if event is None else readout_df['charge_adc_words'][event]
        for i in range(0, 32):
            if len(find_peaks(charge_channels[i,:], height=10, prominence=8)[0]) > 0 or len(find_peaks(-charge_channels[i,:], height=10, prominence=8)[0]) > 0:
                return True 
            if np.any(charge_channels[i,:] < 100):
                return True
        return False

    def find_hits(readout_df, event, height=10, prom=15):
        """
        Find the channel and peak of charge hits in an event across 32 channels.
        :param readout_df: pandas Dataframe
        :param event: int Event number
        :param height: int The height requirement see scipy.signal.find_peaks
        :param prom: int The prominence requirement scipy.signal.find_peaks
        :return:
        """
        hit_idx_list = []
        hit_channel_list = []
        for i in range(0, 32):
            if readout_df['charge_adc_words'][event].ndim < 2:
                continue
            idx_arr = find_peaks(readout_df['charge_adc_words'][event][i,:], height=height, prominence=prom)[0]
            if len(idx_arr) > 0:
                hit_idx_list.append(idx_arr)
                hit_channel_list.append(i)

        return np.asarray(hit_channel_list)

    def get_light_trigger_sample(trig_frame, trig_sample, light_frames, light_samples):
        """
        When using the light as a trigger we can find the light sample closest to the trigger sample
        and use that as the trigger sample time. The advantage is the light sample is higher resolution
        at 64MHz while the trigger sample is at 2MHz. 
        !This does not work for an external trigger.
        """
        light_frames[light_frames == trig_frame]
        # Get all the ROIs in the trigger frame
        light_samples_in_trig_frame = light_samples[light_frames == trig_frame]
        #Convert the trigger sample from 2MHz to 64MHz
        trig_sample_64mhz = trig_sample * 32
        # Find the nearest light ROI to the trigger sample, this should be the one which caused the trigger
        nearest_roi_idx = np.argmin(np.abs(trig_sample_64mhz - light_samples_in_trig_frame))
        return light_samples_in_trig_frame[nearest_roi_idx]

    def get_full_light_data(readout_df, event, channel):
        """
        Reconstruct the full 4 light frames in an event given the ROIs.
        """
        fem_number = np.arange(len(readout_df["slot_number"][event]))[readout_df["slot_number"][event] == 16]
        min_light_frame = np.min(readout_df['light_frame_number'][event]).astype(float)

        light_trigger_sample = get_light_trigger_sample(trig_frame=readout_df['trigger_frame_number'][event][fem_number].astype(float), 
                                                          trig_sample=readout_df['trigger_sample'][event][fem_number].astype(float), 
                                                          light_frames=readout_df['light_frame_number'][event].astype(float), 
                                                          light_samples=readout_df['light_readout_sample'][event].astype(float))

        full_waveform = decoder_bindings.get_full_light_waveform(channel, readout_df['light_channel'][event].astype(int), 
                                                                 min_light_frame,
                                                                 readout_df['light_readout_sample'][event].astype(float), 
                                                                 readout_df['light_frame_number'][event].astype(float), 
                                                                 readout_df['light_adc_words'][event], 255)

        full_axis = decoder_bindings.get_full_light_axis(readout_df['trigger_frame_number'][event][fem_number].astype(float), 
                                                          light_trigger_sample, 
                                                          min_light_frame, 255, True)

        return full_axis, full_waveform
    return are_hits, find_hits


@app.cell
def _(mo):
    mo.md(r"""
    ## Load raw binary data
    """)
    return


@app.cell
def _(get_raw_data, np):
    # /NAS/ColumbiaIntegration/
    num_files = 1
    run_number = '819'

    files = []
    for i in np.arange(num_files):
        files.append(f"/NAS/ColumbiaIntegration/readout_data/pGRAMS_bin_{run_number}_{i}.dat")

    use_charge_roi = False
    readout_df = get_raw_data.get_event_data(files=files, light_slot=16, use_charge_roi=use_charge_roi, channel_threshold=[2055]*192)
    return (readout_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preview Events and Hits
    """)
    return


@app.cell
def _(np, plt):
    def plot_charge_waveforms(event_df, channels, timesize=255, overlay=False, range=[], create_fig=True, show_legend=True):
        """
        Show the charge waveforms for an event as a heatmap.
        :param event_df: pandas DataFrame Dataframe with the events
        :param channel_range: tuple Contiguous charge channel range
        :param timesize: int Readout window size in 2MHz ticks
        :param overlay: bool Show each channel offset in y by ascending order
        :param range: tuple Plot yaxis range
        :param create_fig: bool Create the figure locally, if false use an existing figure object
        :param show_legend: bool Show the plot legend
        :return:
        """
        using_charge_roi = len(event_df['charge_adc_idx']) > 0

        # baseline_subtract = 2048
        baseline_subtract = 0
        # fig_height = 18 if (ch_up - ch_down) > 15 else 12
        # if ch_down == ch_up or overlay:
        #     baseline_subtract = 0
        #     fig_height = 8

        if create_fig: plt.figure(figsize=(18, 4))
        for i, qch in enumerate(event_df['charge_adc_words'][channels,:]):
            baseline_shift = i * 2000 if not overlay else 0
            if using_charge_roi:
                xdown, xup = event_df['charge_adc_idx'][i][0], event_df['charge_adc_idx'][i][-1]+1
                xaxis = 0.5 * (np.linspace(xdown, xup, len(qch)) - timesize) # 2Mhz ticks
            else:
                xaxis = np.linspace(-timesize / 2, timesize, len(qch)) # 2Mhz ticks
            plt.plot(xaxis, qch.astype(int) - baseline_subtract + baseline_shift, label=f'Channel {channels[i]}')

        plt.xlim(-timesize / 2, timesize)
        plt.ylim(range[0], range[1])
        plt.xticks(np.arange(-int(timesize / 10) * 10, int((timesize * 2) / 10) * 10, 50))
        plt.title("Event " + str(event_df['event_index']))
        plt.xlabel("[$\\mu$s]")
        plt.ylabel("Charge [ADC]")
        if show_legend: plt.legend(loc='best')
        # if ch_down == ch_up or overlay: plt.ylim(range)
        if create_fig: plt.show()
    return (plot_charge_waveforms,)


@app.cell
def _(
    are_hits,
    defaultdict,
    find_hits,
    plot_charge_waveforms,
    plt,
    readout_df,
):
    """
    Plotting of both the charge and light signals.
    """
    light_channel = 0
    channel_range = [0,30]
    select_hit_events = True
    bypass = False # bypass hitfinding and look at waveforms
    counter = 0

    peaks = defaultdict(list)
    peak_times = defaultdict(list)

    for event in range(0,len(readout_df)):

        if are_hits(readout_df=readout_df, event=event) or bypass:
            # modified to EXCLUDE events that have hits (including baseline drop out)
            # if are_hits(readout_df=readout_df, event=event) and select_hit_events:
            hit_channels = find_hits(readout_df, event, height=5, prom=7)
            if len(hit_channels)>0 or bypass:
                _fig, ax = plt.subplots(figsize=(16,4))
                plot_charge_waveforms(event_df=readout_df.iloc[event], channels=hit_channels.astype(int), timesize=255, overlay=True, range=[2040, 2080], create_fig=False, show_legend=True)
                # plot_charge_waveforms(event_df=readout_df.iloc[event], channels=np.arange(channel_range[0], channel_range[1]), timesize=255, overlay=True, range=[1800, 2100], create_fig=False, show_legend=True)

                # extract peak time of waveforms
                # for ch in hit_channels:
                #     wf = readout_df.iloc[event]['charge_adc_words'][ch]
                #     peak = np.max(wf)
                #     peak_time = np.argmax(wf)

                #     peaks[ch].append(peak)
                #     peak_times[ch].append(peak_time)


                plt.xlabel("[$\\mu$s]")
                # plt.axvline(-128, color='red', linestyle='--')
                # plt.axvline(0, color='red', linestyle='--')
                plt.xlim(-130,260)
                plt.minorticks_on()
                plt.show()
                counter+=1
        else:
            continue
    return


@app.cell
def _(defaultdict, find_peaks, np):
    def charge_stats(readout_df, channels, filter):
        # stats[channel_id]['metric_name'] = [list_of_values]
        stats = {ch: defaultdict(list) for ch in channels}

        for event in range(len(readout_df)):
            # Get the dictionary of waveforms for this event
            all_waveforms = readout_df.iloc[event].get('charge_adc_words', {})

            for ch in channels:
                wf = all_waveforms[ch]

                if filter:
                    peaks_found, _ = find_peaks(wf, height=5, prominence=7)
                    if len(peaks_found) == 0:
                        continue

                # peak amplitude and position
                peak_val = np.max(wf)
                peak_pos = np.argmax(wf)

                # rise time calculation 
                rise_time = 0
                if peak_val > 0:
                    threshold = 0.2 * peak_val # time from 10% height to peak position
                    # slice waveform up to the peak to look for the start
                    pre_peak_wf = wf[:peak_pos]
                    if len(pre_peak_wf) > 0:
                        # np.argmax on a boolean array returns the index of the first True
                        start_pos = np.argmax(pre_peak_wf > threshold)
                        rise_time = peak_pos - start_pos

                # Append data
                stats[ch]['peaks'].append(peak_val)
                stats[ch]['times'].append(peak_pos)
                stats[ch]['rise'].append(rise_time * 0.5)

        return stats
    return (charge_stats,)


@app.cell
def _(charge_stats, np, plt, readout_df):
    stats = charge_stats(readout_df, [4,5,10,13,20,21,26,27], True)

    channels = sorted(list(stats.keys()))

    _fig, _axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 6), sharey=False)
    ax_flat = _axes.flatten()

    _fig.suptitle("Charge Waveform Peak Amplitudes", fontsize=20)

    _plotrange = [2040, 2070]

    for _i, _ch in enumerate(channels):
        _ax = ax_flat[_i]

        _binning = np.linspace(_plotrange[0], _plotrange[1], 32)

        _ax.hist(stats[_ch]['peaks'], bins=_binning,color='blue', alpha=0.7)
        _ax.set_title(f"Channel {_ch}")
        _ax.set_xlabel("Amplitude (ADC Counts)")
        _ax.grid(True, alpha=0.3)
        _ax.semilogy()
        _ax.axvline(x=2048, color='r', linestyle='--', lw=1, alpha=0.7, label='2048 ADC')
        _ax.set_xlim(_plotrange[0], _plotrange[1])
        _ax.legend()

    plt.tight_layout()
    plt.show()
    return channels, stats


@app.cell
def _(channels, np, plt, stats):
    _fig, _axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 6), sharey=False)
    _ax_flat = _axes.flatten()

    _fig.suptitle("Peak Positions in Time", fontsize=16)

    _plotrange = [0, 800]

    for _i, _ch in enumerate(channels):
        _ax = _ax_flat[_i]

        _binning = np.linspace(_plotrange[0], _plotrange[1], 35)

        _ax.hist(stats[_ch]['times'], bins=_binning, color='tab:orange', alpha=0.7)
        _ax.set_title(f"Channel {_ch}")
        _ax.set_xlabel("Time (ticks)")
        _ax.grid(True, alpha=0.3)
        _ax.semilogy()
        _ax.axvline(x=255, color='r', linestyle='--', lw=1, alpha=0.7, label='Trigger Time')
        _ax.set_xlim(_plotrange[0], _plotrange[1])
        _ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(channels, np, plt, stats):
    _fig, _axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 6), sharey=False)
    _ax_flat = _axes.flatten()

    _fig.suptitle("Charge Waveform Rise Times", fontsize=16)

    _plotrange = [0, 100]

    for _i, _ch in enumerate(channels):
        _ax = _ax_flat[_i]

        _binning = np.linspace(_plotrange[0], _plotrange[1], 35)

        _ax.hist(stats[_ch]['rise'], bins=_binning, color='tab:green', alpha=0.7)
        _ax.set_title(f"Channel {_ch}")
        _ax.set_xlabel("Rise Time (ticks)")
        _ax.grid(True, alpha=0.3)
        _ax.semilogy()
        _ax.set_xlim(_plotrange[0], _plotrange[1])

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    neu_channels = ["B13", "A14", "B12", "A13", "B11", "A12", "B10", "A11", "B9", "A10", "B8", "A9", "B7", "A8", "B6", "A7", "B5", "A6", "B4", "A5", "B3", "A4", "B2", "A3", "B1", "A2", "B0", "A1", "GND", "A0"]
    return


@app.cell
def _(readout_df):
    x = [29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
    y = [26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0]


    event_idx = 368
    event_data= readout_df.iloc[event_idx]
    data = event_data['charge_adc_words']
    return data, event_idx, x, y


@app.cell
def _(data, event_idx, np, plt, x, y):
    x_channels = data[x, 256:-1]
    drift_samples = x_channels.shape[1]
    scale = 0.5

    y_channels = data[y, 256:-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im = ax1.imshow(
        x_channels.T,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[0, x_channels.shape[0], 0, drift_samples * scale]  # x_min, x_max, y_min, y_max
    )
    plt.colorbar(im, ax=ax1, label='Amplitude (ADC)')
    ax1.set_title(f'Event {event_idx} XZ')
    ax1.set_ylabel('Drift (µs)')
    ax1.set_xlabel('X Channel')
    ax1.set_xticks(ticks=np.arange(x_channels.shape[0]))
    # ax1.set_ylim(0,70)

    _im = ax2.imshow(
        y_channels.T,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        extent=[0, y_channels.shape[0], 0, drift_samples * scale]
    )
    plt.colorbar(_im, ax=ax2, label='Amplitude (ADC)')
    ax2.set_title(f'Event {event_idx} YZ')
    ax2.set_ylabel('Drift (µs)')
    ax2.set_xlabel('Y Channel')
    ax2.set_xlim(0,15)
    # ax2.set_ylim(0,70)
    ax2.set_xticks(ticks=np.arange(y_channels.shape[0]+1))

    plt.tight_layout()
    plt.show()
    return x_channels, y_channels


@app.cell
def _():
    import plotly.graph_objects as go
    return (go,)


@app.cell
def _(event_idx, go, np, x_channels, y_channels):
    # Drift window and display options
    drift_idx_range = (0, 763)  # indices into drift samples
    charge_thresh = 2058             # threshold on averaged charge
    xrange = (0, 14)              # x index range (A0..A14 -> 0..14)
    yrange = (0, 13)              # y index range (B0..B13 -> 0..13)
    drift_range = drift_idx_range # z-axis range to match slice



    # Optional: baseline subtraction (uncomment if desired)
    # baseline = 2048
    # xchannels = xchannels.astype(np.int32) - baseline
    # ychannels = ychannels.astype(np.int32) - baseline


    def ugrams_event_display(xchannels, ychannels, charge_thresh,
                             drift_idx_range=(250,300),
                             xrange=(0, 15), yrange=(0, 15), drift_range=(250, 300)):
        """
        Display events that have had the channels mapped
        :param xchannels: 2D array Xaxis of the display (num_x_channels, drift_samples)
        :param ychannels: 2D array Yaxis of the display (num_y_channels, drift_samples)
        :param charge_thresh: int Threshold selecting which points are displayed
        :param drift_idx_range: tuple The index limits for the drift axis
        :param xrange: tuple
        :param yrange: tuple
        :param drift_range: tuple
        """
        if xchannels.ndim != 2 or ychannels.ndim != 2:
            raise ValueError("x_channels and y_channels must both be 2D, shape=(num_channles, drift_samples)")
        if xchannels.shape[1] != ychannels.shape[1]:
            raise ValueError("xchannels and ychannels must have same number of drift samples")

        t0, t1 = drift_idx_range
        drift_slice = slice(t0, t1)

        x_coords = []
        y_coords = []
        z_coords = []
        charge_values = []

        # Iterate through each drift index slice and build the XY plane
        # Note: xchannels[:, drift_slice].T has shape (T, num_x); same for y
        for z_rel, (xch, ych) in enumerate(zip(xchannels[:, drift_slice].T, ychannels[:, drift_slice].T)):
            # xch: length num_x, ych: length num_y
            xplane, yplane = np.meshgrid(xch, ych)              # shapes (num_y, num_x)
            charge = 0.5 * (xplane + yplane)                    # averaged charge

            # Threshold mask
            mask = charge > charge_thresh
            if not np.any(mask):
                continue


            ys, xs = np.where(mask)                             # y-index over ychannels, x-index over xchannels
            x_coords.extend(xs.tolist())
            y_coords.extend(ys.tolist())
            z_abs = z_rel + t0
            z_coords.extend([z_abs] * len(xs))
            charge_values.extend(charge[mask].ravel().tolist())

        fig = go.Figure(data=[go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=4,
                color=charge_values,
                colorscale='Plasma',
                colorbar=dict(title='Waveform Amplitude'),
                opacity=0.85,
                showscale=True
            )
        )])

        fig.update_layout(
            title=f'uGRAMS Event (event_idx={event_idx})',
            scene=dict(
                xaxis_range=xrange,
                yaxis_range=yrange,
                zaxis_range=drift_range,
                xaxis_title='X: A channels (index A0→0 ... A14→14)',
                yaxis_title='Y: B channels (index B0→0 ... B13→13)',
                zaxis_title='Z: Drift (sample index)'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig.show()

    # ---------------------------
    # Render
    # ---------------------------
    ugrams_event_display(
        xchannels=x_channels,
        ychannels=y_channels,
        charge_thresh=charge_thresh,
        drift_idx_range=drift_idx_range,
        xrange=xrange,
        yrange=yrange,
        drift_range=drift_range
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
