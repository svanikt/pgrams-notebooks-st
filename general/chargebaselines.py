import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""## November 2025 Integration Tests: Charge Baselines""")
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
    colors = sns.color_palette('colorblind')


    # apply sexy Latex matplotlib style
    plt.style.use('/home/pgrams/latex-cm.mplstyle')


    # Add the location of the data utilities code to the system path
    # Can also just add it to PYTHONPATH but make sure you know where it is pointing
    # or you could end up using the wrong code.

    # The root directory for the utility code is up 2 directories from the notebook
    abs_repo_path = os.path.abspath('../../')
    # Insert the path to the front of sys.path if it is not already there
    if not abs_repo_path in sys.path:
        sys.path.insert(0, abs_repo_path)
    return colors, curve_fit, find_peaks, mo, np, pd, plt


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

    def plot_charge_waveforms(readout_df, event, channel_range, timesize=255, overlay=False, range=[], create_fig=True, show_legend=True):

        ch_down = channel_range[0]
        ch_up = channel_range[1]

        baseline_subtract = 2000
        fig_height = 18 if (ch_up - ch_down) > 15 else 12
        if ch_down == ch_up or overlay:
            baseline_subtract = 0
            fig_height = 8

        if create_fig: plt.figure(figsize=(18,fig_height))
        for i, qch in enumerate(readout_df['charge_adc_words'][event][ch_down:ch_up+1,:]):
            baseline_shift = i * 2000 if not overlay else 0
            plt.plot(np.linspace(-timesize/2, timesize, len(qch)), qch.astype(int) - baseline_subtract + baseline_shift, label=f'Channel {ch_down+i}')

        xdown, xup = -int(timesize/2), timesize
        plt.xticks(np.arange(-int(timesize/10)*10, int((timesize*2)/10)*10, 50))
        plt.title("Event " + str(event))
        plt.xlabel("[$\\mu$s]")
        plt.ylabel("Charge [ADC]")
        if show_legend: plt.legend(loc='best')
        if ch_down == ch_up or overlay: plt.ylim(range)
        if create_fig: plt.show()


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
    return are_hits, gaussian, plot_charge_waveforms


@app.cell
def _(mo):
    mo.md(r"""## Load raw binary data""")
    return


@app.cell
def _(get_raw_data, np):

    num_files = 1
    run_number = '715'

    files = []
    for i in np.arange(num_files):
        files.append(f"/home/pgrams/data/jan13_integration/readout_data/pGRAMS_bin_{run_number}_{i}.dat")

    use_charge_roi = False
    readout_df = get_raw_data.get_event_data(files=files, light_slot=16, use_charge_roi=use_charge_roi, channel_threshold=[2055]*192)
    return num_files, readout_df, run_number


@app.cell
def _(readout_df):
    readout_df.tail(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Preview Events""")
    return


@app.cell
def _(are_hits, plot_charge_waveforms, plt, readout_df):
    """
    Plotting of both the charge and light signals.
    """
    light_channel = 0
    channel_range = [0,31]
    # channel_range = [32,63]
    # channel_range = [64,95]
    # channel_range = [96,127]
    # channel_range = [128, 159]
    # channel_range = [160, 191]

    select_hit_events = False

    for event in range(0,10):
        if readout_df['charge_adc_words'][event].ndim < 2:
            continue
        # if not are_hits(readout_df=readout_df, event=event) and select_hit_events:
        #     continue

        # modified to EXCLUDE events that have hits (including baseline drop out)
        if are_hits(readout_df=readout_df, event=event) and select_hit_events:
            continue

        fig, ax1 = plt.subplots(figsize=(16,4))
        plot_charge_waveforms(readout_df=readout_df, event=event, channel_range=channel_range, timesize=255, overlay=True, range=[2000, 2100], create_fig=False, show_legend=False)

        # ax1.set_ylim(2043, 2051)

        ax2 = ax1.twinx()
        # ax2.plot(full_axis/1e3, full_waveform-0, color='red', linestyle='--')
        ax2.set_ylim(2000, 2500)
        ax2.set_ylabel('Light [ADC]')

        plt.xlabel("[$\\mu$s]")
        # plt.axvline(-128, color='red', linestyle='--')
        # plt.axvline(0, color='red', linestyle='--')
        plt.xlim(-130,260)
        plt.minorticks_on()
        plt.show()
    return channel_range, select_hit_events


@app.cell
def _(are_hits, channel_range, np, readout_df, select_hit_events):
    counts = {ch: [] for ch in range(channel_range[0], channel_range[1])}
    for evt in range(0,len(readout_df)):
        if readout_df['charge_adc_words'][evt].ndim < 2:
            continue
        # modified to EXCLUDE events that have hits (including drop out events)
        if are_hits(readout_df=readout_df, event=evt) and select_hit_events:
            continue

        for ch in np.arange(channel_range[0],channel_range[1]):
            counts[ch].append(readout_df['charge_adc_words'][evt][ch,0:125])
    return (counts,)


@app.cell
def _(colors, curve_fit, file_list, gaussian, np, pd, plt, savedir):
    def noise_hist(counts, channels=np.arange(0,30), range=20, num_bins=40, subplot=10, saveas=''):
        # based on input range and desired num_bins, create bins
        bins = np.linspace(2048 - (range/2), 2048 + (range/2), num_bins)
        baselines = {} 
        rms = {}

        # if there is only one input channel, automatically turn off subplot
        if len(channels)==1: subplot=0

        # if subplot is enabled, find number of rows necessary and initialize plot
        if subplot:
            rows = len(channels) // subplot + (len(channels) % subplot > 0)
            if len(channels)>=subplot: 
                _, axs = plt.subplots(rows, subplot, figsize=(20, 2*rows))
            else: 
                _, axs = plt.subplots(rows, len(channels), figsize=(20, 2*len(channels)))

        for i, ch in enumerate(channels):
            if subplot:
                row = i//subplot # row index
                col = i%subplot # column index
                ax = axs[row, col]
            else:
                plt.figure(figsize=(7, 6))
                ax = plt.gca()

            all_counts = np.concatenate(counts[ch])

            # compute mean and std based on raw data
            mean = np.mean(all_counts)
            std = np.std(all_counts)

            # create histogram for current channel, find bin midpoints
            hist, bin_edges = np.histogram(all_counts, bins=bins, density=True)
            midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            # fit gaussian to midpoints, initial guess based on mean and std of counts
            params, _ = curve_fit(gaussian, midpoints, hist, [np.mean(all_counts), np.std(all_counts)])
            mean_fit, std_fit = params

            # trust the fitted result more, so we return this in the dictionary
            baselines[ch] = (mean_fit)
            rms[ch] = std_fit

            # plot histogram
            ax.hist(all_counts, bins=bins, histtype='step', color=colors[0], density=True)

            # plot fitted gaussian
            x = np.linspace(min(bins), max(bins), range)
            fit = gaussian(x, mean, std)
            ax.fill_between(x, 0, fit, color=colors[9], alpha=0.5, label='Gaussian Fit')

            # plot expected and actual baseline positions
            ax.axvline(2048, color='r', ls='--', alpha=0.5, label='Expected BL')   
            ax.axvline(mean, color='r', ls='--', label='Actual BL')

            # title and legend
            ax.set_title(f'Ch. {ch}')

            ax.text(
                0.98, 0.95,
                r"$\bf{Raw:}$" + f"\nBL={mean:.1f}\nRMS={std:.1f}\n"
                r"$\bf{Gaussian:}$" + f"\nBL={mean_fit:.1f}\nRMS={std_fit:.1f}",
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )


            # ax.legend()

        plt.tight_layout()
        if saveas: plt.savefig(f'{savedir}/BaselineCountsRun{file_list}.{saveas}', dpi=400, bbox_inches='tight') # save plot

        # tabulate and display results
        results = {
            "Ch": list(baselines.keys()),
            "Mean Counts": list(baselines.values()),
            "RMS Counts": list(rms.values())
        }
        results_df = pd.DataFrame(results)
        # display(results_df)

        return baselines, rms
    return (noise_hist,)


@app.cell
def _(counts, noise_hist, np, plt):
    # baselines, rms= noise_hist(counts, channels=np.arange(32,62), range=100, num_bins=100, subplot=6)
    baselines, rms= noise_hist(counts, channels=np.arange(0,30), range=100, num_bins=100, subplot=6)
    plt.show()
    return baselines, rms


@app.cell
def _(baselines, colors, plt):
    channels = list(baselines.keys())
    baseline = [baselines[ch] for ch in channels]
    neu_channels = ["B13", "A14", "B12", "A13", "B11", "A12", "B10", "A11", "B9", "A10", "B8", "A9", "B7", "A8", "B6", "A7", "B5", "A6", "B4", "A5", "B3", "A4", "B2", "A3", "B1", "A2", "B0", "A1", "GND", "A0"]

    figure, ax = plt.subplots(figsize=(10, 5))
    ax.bar(channels, baseline, color=colors[9], alpha=0.4)
    ax.set_xlabel('CU ADC Channel', fontsize=14)
    ax.set_ylabel('Baseline (ADC Counts)', fontsize=14)
    ax.set_xticks(channels)
    ax.set_xticklabels([str(ch) for ch in channels], rotation=90, fontsize=10)

    # Add secondary x-axis
    ax3 = ax.twiny()
    ax3.set_xlim(ax.get_xlim())
    ax3.set_xticks(channels)
    ax3.set_xticklabels([str(ch) for ch in neu_channels], rotation=90, fontsize=10)
    ax3.set_xlabel('NEU TPC Channel', fontsize=14)

    plt.ylim(2030, 2060)
    plt.tight_layout()
    plt.show()
    return baseline, channels, neu_channels


@app.cell
def _(channels, colors, neu_channels, np, plt, rms, run_number):
    noise_rms = [rms[ch] for ch in channels]

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.bar(channels, noise_rms, color=colors[2], alpha=0.4)
    _ax.set_xlabel('CU ADC Channel', fontsize=14)
    _ax.set_ylabel('Standard Deviation (ADC Counts)', fontsize=14)
    _ax.set_xticks(channels)
    _ax.set_xticklabels([str(ch) for ch in channels], rotation=90, fontsize=10)

    # Add secondary x-axis
    _ax2 = _ax.twiny()
    _ax2.set_xlim(_ax.get_xlim())
    _ax2.set_xticks(channels)
    _ax2.set_xticklabels([str(ch) for ch in neu_channels], rotation=90, fontsize=10)
    _ax2.set_xlabel('NEU TPC Channel', fontsize=14)
    _ax2.text(channels[-1]-6, np.max(noise_rms) - (np.ptp(noise_rms)*0.05), f'R{run_number}: Q1 Bottom Connector')
    plt.tight_layout()
    plt.show()
    return (noise_rms,)


@app.cell
def _(get_raw_data, np, num_files):
    # overlay ADC only noise
    # /NAS/ColumbiaIntegration/
    _num_files = 1
    _run_number = '458' # ADC only runs

    _files = []
    for _i in np.arange(num_files):
        _files.append(f"/NAS/ColumbiaIntegration/readout_data/pGRAMS_bin_{_run_number}_{_i}.dat")

    _use_charge_roi = False
    readout_df1 = get_raw_data.get_event_data(files=_files, light_slot=16, use_charge_roi=_use_charge_roi, channel_threshold=[2055]*192)
    return (readout_df1,)


@app.cell
def _(are_hits, channel_range, np, readout_df1, select_hit_events):
    counts1 = {_ch: [] for _ch in range(channel_range[0], channel_range[1])}
    for _event in range(0,1000):
        if readout_df1['charge_adc_words'][_event].ndim < 2:
            continue
        # modified to EXCLUDE events that have hits (including drop out events)
        if are_hits(readout_df=readout_df1, event=_event) and select_hit_events:
            continue


        for _ch in np.arange(channel_range[0],channel_range[1]):
            counts1[_ch].append(readout_df1['charge_adc_words'][_event][_ch,:])
    return (counts1,)


@app.cell
def _(counts1, noise_hist, np, plt):
    baselines1, rms1= noise_hist(counts1, channels=np.arange(0,30), range=100, num_bins=100, subplot=6)
    plt.show()
    return baselines1, rms1


@app.cell
def _(
    baseline,
    baselines1,
    channels,
    colors,
    neu_channels,
    noise_rms,
    np,
    plt,
    rms1,
):
    baseline1 = [baselines1[ch] for ch in channels]
    noise_rms1 = [rms1[ch] for ch in channels]


    bar_width = 0.4
    x = np.arange(len(channels))

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.grid(which='both', visible='True', lw=0.2, ls='--', alpha=0.5)

    # Plot the first bar set, shifted left
    bars1 = _ax.bar(x + bar_width/2, baseline, width=bar_width, color=colors[9], alpha=0.4, label='Full Readout Chain',
    yerr=noise_rms, capsize=2, error_kw=dict(ecolor='royalblue', alpha=0.7,  elinewidth=1, capthick=1))
    # Plot the second bar set, shifted right
    bars2 = _ax.bar(x - bar_width/2, baseline1, width=bar_width, color=colors[9], alpha=0.7, label='ADC Only',
    yerr=noise_rms1, capsize=2, error_kw=dict(ecolor='royalblue', alpha=0.7, elinewidth=1, capthick=1))

    _ax.set_xlabel('CU ADC Channel', fontsize=8)
    _ax.set_ylabel('Baseline (ADC Counts)', fontsize=14)
    _ax.set_xticks(x)
    _ax.set_xticklabels([str(ch) for ch in channels], rotation=90, fontsize=10)

    # Add legend
    _ax.legend(fontsize=12)


    # Add secondary x-axis
    _ax2 = _ax.twiny()
    _ax2.set_xlim(_ax.get_xlim())
    _ax2.set_xticks(x)
    _ax2.set_xticklabels([str(ch) for ch in neu_channels], rotation=90, fontsize=10)
    _ax2.set_xlabel('NEU TPC Channel', fontsize=8)
    plt.tight_layout()

    # plt.title('Baselines+RMS -- TPC HV on')

    plt.ylim(2030, 2060)

    plt.show()
    return bar_width, noise_rms1, x


@app.cell
def _(
    bar_width,
    channels,
    colors,
    neu_channels,
    noise_rms,
    noise_rms1,
    plt,
    x,
):
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.grid(which='both', visible='True', lw=0.2, ls='--', alpha=0.5)

    # Plot the first bar set, shifted left
    _bars1 = _ax.bar(x + bar_width/2, noise_rms, width=bar_width, color=colors[2], alpha=0.4, label='Full Readout Chain')
    # Plot the second bar set, shifted right
    _bars2 = _ax.bar(x - bar_width/2, noise_rms1, width=bar_width, color=colors[2], alpha=0.7, label='ADC Only')

    _ax.set_xlabel('CU ADC Channel', fontsize=14)
    _ax.set_ylabel('Noise RMS (ADC Counts)', fontsize=14)
    _ax.set_xticks(x)
    _ax.set_xticklabels([str(ch) for ch in channels], rotation=90, fontsize=10)

    # Add legend
    _ax.legend(fontsize=12)

    # Add secondary x-axis
    _ax2 = _ax.twiny()
    _ax2.set_xlim(_ax.get_xlim())
    _ax2.set_xticks(x)
    _ax2.set_xticklabels([str(ch) for ch in neu_channels], rotation=90, fontsize=10)
    _ax2.set_xlabel('NEU TPC Channel', fontsize=14)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
