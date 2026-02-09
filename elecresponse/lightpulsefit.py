import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import decoder_bindings
    import os
    import sys 
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # from IPython import display
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    from scipy.integrate import trapezoid
    from prettytable import PrettyTable
    import seaborn as sns
    colors = sns.color_palette('colorblind')

    # Add the location of the data utilities code to the system path
    # Can also just add it to PYTHONPATH but make sure you know where it is pointing
    # or you could end up using the wrong code.

    # The root directory for the utility code is up 2 directories from the notebook
    abs_repo_path = os.path.abspath('/home/pgrams/daq_analysis/PGramsRawData')
    # Insert the path to the front of sys.path if it is not already there
    if not abs_repo_path in sys.path:
        sys.path.insert(0, abs_repo_path)

    plt.style.use('/home/pgrams/latex-cm.mplstyle')
    return (
        PrettyTable,
        colors,
        curve_fit,
        decoder_bindings,
        find_peaks,
        make_axes_locatable,
        mo,
        np,
        pd,
        plt,
        trapezoid,
    )


@app.cell
def _():
    import raw_data_ana.get_data as get_raw_data
    import raw_data_ana.data_checks as data_checks
    import raw_data_ana.plotting as plot
    import raw_data_ana.charge_utils as qutils
    import raw_data_ana.light_utils as lutils

    return get_raw_data, plot


@app.cell
def _(curve_fit, decoder_bindings, find_peaks, np, pd, plt, readout_df):
    def plot_charge_channels(adc_words, event, num_channel, timesize, charge_range=[1950, 2150]):
        xdown, xup = (-int(timesize / 2), timesize)
        plt.figure(figsize=(18, 6))
        plt.imshow(adc_words, cmap=plt.cm.RdBu_r, extent=[xdown, xup, 0, num_channel], vmin=charge_range[0], vmax=charge_range[1], origin='lower')
        plt.plot([0, 0], [0, num_channel], linestyle='--', color='gray')
        plt.plot([xup / 2, xup / 2], [0, num_channel], linestyle='--', color='gray')
        clb = plt.colorbar()
        clb.set_label('Charge  [ADC]')
        plt.xticks(np.arange(-150, 300, 50))
        plt.yticks(np.arange(0, num_channel + 1, 16))
        plt.title('Event ' + str(event))
        plt.xlabel('[$\\mu$s]')
        plt.ylabel('Channel')
        plt.xlim(xdown, xup)
        plt.show()

    def plot_difference(evt, channel):
        shift = 4
        approx_baseline = readout_df['light_adc_words'][evt][channel, 0]
        diff = readout_df['light_adc_words'][evt][channel, shift:].astype(float) - readout_df['light_adc_words'][evt][channel, :-shift].astype(float)
        diff[diff < 0] = 0
        plt.plot(np.concatenate((np.ones(shift) * approx_baseline, diff + approx_baseline)), linestyle='--')

    def gaussian(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def gaussian_exp(x, amplitude, mean, stddev, tau):
        exp = np.exp(-(x - mean) / tau)
        exp[x <= mean] = 0
        gauss = amplitude * (np.exp(-((x - mean) / (2 * stddev)) ** 2) + exp)
        gauss[x > mean] = gauss[x > mean] / 2
        return gauss

    def fit_histogram(h, cb, frac=0.3):
        bc = (cb[:-1] + cb[1:]) / 2
        mask = h > frac * np.max(h)
        popt, pcov = curve_fit(gaussian, bc[mask], h[mask], p0=[max(h), 2000, 200])
        return (bc[mask], popt)

    def plot_trigger_rate(df, fem_num, bin_range, framesize=255, axis=None):
        """
        Reconstruct trigger rate  # popt, pcov = curve_fit(gaussian_exp, bc[mask], h[mask], p0=[max(h), 2050, 25,1])

        Time between trigger is,
        Δs = trig_sample[i] - trig_sample[i-1]
        Δf = trig_frame[i] - trig_frame[i-1]
        framesize = trigger framesize in 2MHz ticks, likely 255 = 128μs for pGRAMS
        ΔT = Δf * framesize + Δs
        """
        f1 = df['event_frame_number'].str[fem_num][:-1].values.astype(float)
        f2 = df['event_frame_number'].str[fem_num][1:].values.astype(float)
        s1 = df['trigger_sample'].str[fem_num][:-1].values.astype(float)
        s2 = df['trigger_sample'].str[fem_num][1:].values.astype(float)
        trig_time_diff = (f2 - f1) * framesize + (s2 - s1)
        trig_time_diff[trig_time_diff == 0] = 1e-10
        if axis is None:
            c, _, _ = plt.hist(1.0 / (trig_time_diff * 5e-07), bins=50, range=bin_range)
            plt.xticks(np.arange(bin_range[0], bin_range[1] + 1, 50))
            plt.xlabel('Trigger Rate [Hz]  (from event $\\Delta$t)')
        else:
            c, _, _ = axis.hist(1.0 / (trig_time_diff * 5e-07), bins=50, range=bin_range, alpha=0.8, density=True)
            axis.set_xticks(np.arange(bin_range[0], bin_range[1] + 1, 50))
            axis.set_xlabel('Trigger Rate [Hz]  (from event $\\Delta$t)')
        print('Number of events in histogram:', np.sum(c))
        print(c)
        return 1.0 / (trig_time_diff * 5e-07)

    def concat_row_arrays(row):
        return np.concatenate(row)

    def getevent_data(files, key_list=None, get_hits_only=False):
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
                    if key_list is not None:
                        tmp = {key: tmp[key] for key in key_list}
                    readout_data.append(tmp)
                    evt_cnt = evt_cnt + 1
                except:
                    continue
                if evt_cnt == 5000:
                    break
        readout_df = pd.DataFrame(readout_data)
        return (readout_df, proc)

    def plot_charge_waveforms(readout_df, event, channel_range, timesize=255, overlay=False, range=[], create_fig=True, show_legend=True):
        ch_down = channel_range[0]
        ch_up = channel_range[1]
        baseline_subtract = 2000
        fig_height = 18 if ch_up - ch_down > 15 else 12
        if ch_down == ch_up or overlay:
            baseline_subtract = 0
            fig_height = 8
        if create_fig:
            plt.figure(figsize=(18, fig_height))
        for i, qch in enumerate(readout_df['charge_adc_words'][event][ch_down:ch_up + 1, :]):
            baseline_shift = i * 2000 if not overlay else 0
            plt.plot(np.linspace(-timesize / 2, timesize, len(qch)), qch.astype(int) - baseline_subtract + baseline_shift, label=f'Channel {ch_down + i}', lw=2)
        xdown, xup = (-int(timesize / 2), timesize)
        plt.xticks(np.arange(-int(timesize / 10) * 10, int(timesize * 2 / 10) * 10, 50))
        plt.title('Event ' + str(event))
        plt.xlabel('[$\\mu$s]')
        plt.ylabel('Charge [ADC]')
        if show_legend:
            plt.legend(loc='best')
        if ch_down == ch_up or overlay:
            plt.ylim(range)
        if create_fig:
            plt.show()

    def are_hits(readout_df, event):
        """
        Use for event selection. Check if there are any large-ish charge hits in the event in any of the 32 channels.
        """
        charge_channels = readout_df['charge_adc_words'] if event is None else readout_df['charge_adc_words'][event]
        for i in range(0, 32):
            if len(find_peaks(charge_channels[i, :], height=10, prominence=15)[0]) > 0:
                return True
        return False

    def find_hits(readout_df, event):
        """
        Find the channel and peak of charge hits in an event across 32 channels.
        """
        hit_idx_list = []
        hit_channel_list = []
        for i in range(0, 32):
            idx_arr = find_peaks(readout_df['charge_adc_words'][event][i, :], height=10, prominence=15)[0]
            if len(idx_arr) > 0:
                hit_idx_list.append(idx_arr)
                hit_channel_list.append(i)
        return (np.asarray(hit_channel_list), np.asarray(hit_idx_list))

    def get_light_trigger_sample(trig_frame, trig_sample, light_frames, light_samples):
        """
        When using the light as a trigger we can find the light sample closest to the trigger sample
        and use that as the trigger sample time. The advantage is the light sample is higher resolution
        at 64MHz while the trigger sample is at 2MHz. 
        !This does not work for an external trigger.
        """
        light_frames[light_frames == trig_frame]
        light_samples_in_trig_frame = light_samples[light_frames == trig_frame]
        trig_sample_64mhz = trig_sample * 32
        nearest_roi_idx = np.argmin(np.abs(trig_sample_64mhz - light_samples_in_trig_frame))
        return light_samples_in_trig_frame[nearest_roi_idx]

    def get_full_light_data(readout_df, event, channel):
        """
        Reconstruct the full 4 light frames in an event given the ROIs.
        """
        fem_number = np.arange(len(readout_df['slot_number'][event]))[readout_df['slot_number'][event] == 16]
        min_light_frame = np.min(readout_df['light_frame_number'][event]).astype(float)
        light_trigger_sample = get_light_trigger_sample(trig_frame=readout_df['trigger_frame_number'][event][fem_number].astype(float), trig_sample=readout_df['trigger_sample'][event][fem_number].astype(float), light_frames=readout_df['light_frame_number'][event].astype(float), light_samples=readout_df['light_readout_sample'][event].astype(float))
        full_waveform = decoder_bindings.get_full_light_waveform(channel, readout_df['light_channel'][event].astype(int), min_light_frame, readout_df['light_readout_sample'][event].astype(float), readout_df['light_frame_number'][event].astype(float), readout_df['light_adc_words'][event], 255)
        full_axis = decoder_bindings.get_full_light_axis(readout_df['trigger_frame_number'][event][fem_number].astype(float), light_trigger_sample, min_light_frame, 255, True)
        return (full_axis, full_waveform)

    return gaussian, gaussian_exp, get_light_trigger_sample


@app.cell
def _(get_raw_data, np):
    # /NAS/ColumbiaIntegration/
    num_files = 1
    run_number = '139'

    files = []
    for _i in np.arange(num_files):
        files.append(f"/nevis/riverside/data/jsen/daq_data/neu_06_2025/pGRAMS_bin_{run_number}_{_i}.dat")

    use_charge_roi = True
    readout_df = get_raw_data.get_event_data(files=files, light_slot=16, use_charge_roi=use_charge_roi, channel_threshold=[2055]*192)
    return (readout_df,)


@app.cell
def _(readout_df):
    readout_df
    return


@app.cell
def _(PrettyTable, np, readout_df):
    light_samples_list = []
    light_roi_list = []
    charge_samples_list = []
    charge_channels_list = []
    for law,caw in zip(readout_df['light_adc_words'],readout_df['charge_adc_words']):
        if len(caw) > 0:
            charge_channels_list.append(caw.shape[0])
            charge_samples_list.append(caw.shape[1])
        else:
            charge_channels_list.append(0)
        if len(law) > 0:
            light_samples_list.append(law.shape[1])
            light_roi_list.append(law.shape[0])
        else:
            light_roi_list.append(0)


    table = PrettyTable()
    table.field_names = ["Charge", "Values", "Count"]

    types,cnt = np.unique(charge_channels_list, return_counts=True)
    table.add_row(["Channels per Event", types, cnt])

    types,cnt = np.unique(charge_samples_list, return_counts=True)
    table.add_row(["Words per Channel", types, cnt])
    print(table)

    table = PrettyTable()
    table.field_names = ["Light", "Values", "Count"]

    types,cnt = np.unique(light_roi_list, return_counts=True)
    table.add_row(["ROIs per Event", types, cnt])

    types,cnt = np.unique(light_samples_list, return_counts=True)
    table.add_row(["Words per Channel", types, cnt])

    types,cnt = np.unique(np.concatenate(readout_df['light_channel'].values), return_counts=True)
    table.add_row(["Channels", types, cnt])

    types,cnt = np.unique(np.concatenate(readout_df['light_trigger_id'].values), return_counts=True)
    table.add_row(["Disc ID", types, cnt])

    print(table)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preview Light Events
    """)
    return


@app.cell
def _(plot, readout_df):
    plot.plot_light_waveforms(readout_df, [0,3], ylim=(2000,2080),show_diff=True, show_legend=False, show_events=True)
    return


@app.cell
def _(channel, decoder_bindings, get_light_trigger_sample, np, readout_df):
    fem_number = 16
    timesize = 255
    min_light_frame = np.min(readout_df['light_frame_number'].astype(float))

    light_trigger_sample = get_light_trigger_sample(trig_frame=readout_df['trigger_frame_number'][fem_number].astype(float),
                                                    trig_sample=readout_df['trigger_sample'][fem_number].astype(float),
                                                    light_frames=readout_df['light_frame_number'].astype(float),
                                                    light_samples=readout_df['light_readout_sample'].astype(float))


    waveform = decoder_bindings.get_full_light_waveform(channel, readout_df['light_channel'].astype(int),
                                                                 min_light_frame,
                                                                 readout_df['light_readout_sample'].astype(float),
                                                                 readout_df['light_frame_number'].astype(float),
                                                                 readout_df['light_adc_words'], timesize)

    axis = decoder_bindings.get_full_light_axis(readout_df['trigger_frame_number'][fem_number].astype(float),
                                                         light_trigger_sample,
                                                         min_light_frame, timesize, True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw Baseline
    """)
    return


@app.cell
def _(are_hits_1, np, readout_df):
    _channel_range = [12, 13]
    _select_hit_events = True
    counts = {ch: [] for ch in range(_channel_range[0], _channel_range[1])}
    for _event in range(0, 1000):
        if readout_df['charge_adc_words'][_event].ndim < 2:
            continue
        if not are_hits_1(readout_df=readout_df, event=_event) and _select_hit_events:
            continue
        charge_channels = readout_df['charge_adc_words'] if _event is None else readout_df['charge_adc_words'][_event]
        if np.any(charge_channels[0, :] < 100):
            continue
        for ch in np.arange(_channel_range[0], _channel_range[1]):
            counts[ch].append(readout_df['charge_adc_words'][_event][ch, 0:50])
    return (counts,)


@app.cell
def _(colors, curve_fit, file_list, gaussian, np, pd, plt, savedir):
    def noise_hist(counts, channels=np.arange(0,30), baseline_guess=2048, range=20, num_bins=40, subplot=10, saveas=''):
        # based on input range and desired num_bins, create bins
        bins = np.linspace(baseline_guess - (range/2), baseline_guess + (range/2), num_bins)
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
                plt.figure(figsize=(4, 4))
                ax = plt.gca()

            all_counts = np.concatenate(counts[ch])
            # print(all_counts)

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
            x = np.linspace(min(bins), max(bins), 400)
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

        plt.show()

        # tabulate and display results
        results = {
            "Ch": list(baselines.keys()),
            "Mean Counts": list(baselines.values()),
            "RMS Counts": list(rms.values())
        }
        results_df = pd.DataFrame(results)

        return baselines, rms

    return (noise_hist,)


@app.cell
def _(counts, noise_hist, np):
    baselines, rms= noise_hist(counts, channels=np.arange(12,13), range=50, num_bins=50)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Average + Trim Waveform
    """)
    return


@app.cell
def _(are_hits_1, colors, mo, np, plt, readout_df):
    channel = 1
    desired_length = 763
    sample_interval_us = 0.5
    waveforms = []
    for _event in range(len(readout_df)):
        if readout_df['charge_adc_words'][_event].ndim < 2:
            continue
        if are_hits_1(readout_df=readout_df, event=_event):
            wf = readout_df['charge_adc_words'][_event][channel, :]
            if np.any(wf < 100):
                continue
            if len(wf) >= desired_length:
                waveforms.append(wf[0:desired_length].astype(float))
    waveforms_arr = np.array(waveforms)
    avg_waveform = np.mean(waveforms_arr, axis=0)
    std_waveform = np.std(waveforms_arr, axis=0)
    x_axis = np.linspace(-desired_length / 3, 2 * desired_length / 3, len(avg_waveform)) * sample_interval_us
    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, avg_waveform, color=colors[2])
    plt.fill_between(x_axis, avg_waveform - std_waveform, avg_waveform + std_waveform, color=colors[2], alpha=0.2, label='$\\pm1\\sigma$ contours')
    plt.title(f'Averaged Charge Waveform ({len(readout_df)} events, CSP v7.1)')
    plt.xlabel('Time [$\\mu$s]')
    plt.xlim(-132.5, 260)
    plt.axvline(-127.5, color='black', alpha=0.7, linestyle='--', lw=0.7)
    plt.axvline(254.5, color='black', alpha=0.7, linestyle='--', lw=0.7)
    baseline_region = [0, 230]
    plt.axvline(0.5 * baseline_region[0] - 127.5, color='red', alpha=1, linestyle='--', label='Baseline + RMS Measurement Region')
    plt.axvline(0.5 * baseline_region[1] - 127.5, color='red', alpha=1, linestyle='--')
    baseline = np.mean(avg_waveform[baseline_region[0]:baseline_region[1]])
    rms_1 = np.std(avg_waveform[baseline_region[0]:baseline_region[1]])
    baselinerms = f'\n| Metric | Value |\n|:---|:---:|\n| Baseline | {baseline:.2f} |\n| RMS | {rms_1:.2f} |\n'
    mo.md(baselinerms)
    plt.axhline(baseline, color='#00238A', alpha=0.8, linestyle='--', lw=0.9, label='Measured Baseline')
    plt.ylabel('Charge [ADC]')
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()
    return avg_waveform, baseline, channel, rms_1, waveforms_arr, x_axis


@app.cell
def _(waveforms_arr):
    print(len(waveforms_arr))
    return


@app.cell
def _(avg_waveform, baseline, colors, plt, readout_df, x_axis):
    # invert the waveform so the pulse is a positive peak
    inverted_waveform = baseline - avg_waveform

    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, inverted_waveform, color=colors[2])
    plt.title(f"Baseline Subtracted Avg. Charge Waveform ({len(readout_df)} events)")
    plt.xlabel("Time [$\\mu$s]")
    plt.xlim(-132.5,260)
    plt.axvline(-127.5, color='black', alpha=0.7, linestyle='--')
    plt.axvline(255, color='black', alpha=0.7, linestyle='--')

    # plt.ylim(2035,2050)
    # plt.legend(loc='best')
    plt.ylabel("Charge [ADC]")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()
    return (inverted_waveform,)


@app.cell
def _(inverted_waveform, rms_1, trim_pulse, x_axis):
    trim_pulse(x_axis, inverted_waveform, rms_1, start_pad=-10, end_pad=10, plot=True)
    return


@app.cell
def _(colors, inverted_waveform, np, plt, rms_1, x_axis):
    def trim_pulse(x, y, rms, start_pad=-1, end_pad=0, plot=False):
        """
        Trim the input pulse based on RMS and padding.
        """
        start_idx = np.where(y > 3 * rms)[0][0] + start_pad  # find first index in waveform above 3*rms
        start_idx = max(0, start_idx)
        trimmed_x = x[start_idx:-1]
        trimmed_y = y[start_idx:-1]  # ensure indices are within array bounds
        end_idx = np.where(trimmed_y[-start_pad:] < rms)[0][0] + end_pad
        end_idx = min(len(trimmed_x), end_idx)
        trimmed_x = trimmed_x[0:end_idx]  # slice x and y axes based on these bounds
        trimmed_y = trimmed_y[0:end_idx]
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(trimmed_x, trimmed_y, 'o', markersize=4, alpha=0.8, color=colors[2])  # find first index in waveform below rms (ensure not to include region around starting index)
            plt.title('Trimmed Pulse')
            plt.xlabel('Time [µs]')
            plt.ylabel('Charge [ADC]')
            plt.grid(True)  # slice x and y axes based on these bounds
            plt.show()
        return (trimmed_x, trimmed_y)
    trimmed_x, trimmed_y = trim_pulse(x_axis, inverted_waveform, rms_1, start_pad=-1, end_pad=0, plot=True)  # plot trimmed pulse  # plt.plot(trimmed_x, trimmed_y, '--', color=colors[2], alpha=0.4)
    return trim_pulse, trimmed_x, trimmed_y


@app.cell
def _():
    # np.savetxt("elecpulse.csv", np.vstack((trimmed_x, trimmed_y)).T, delimiter=",", fmt='%.6e')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fits
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ex-Gaussian Fit
    """)
    return


@app.cell
def _(colors, curve_fit, gaussian_exp, np, plt, trimmed_x, trimmed_y):
    # intial guesses for fit parameters
    amp_guess = np.max(trimmed_y)  # peak amplitude
    mean_guess = 8  # time of the peak
    stddev_guess = 2.0  # narrow gaussian part
    tau_guess = 15.0  # exponential tail decay time
    p0 = [amp_guess, mean_guess, stddev_guess, tau_guess]
    _params, _cov = curve_fit(gaussian_exp, trimmed_x, trimmed_y, p0=p0)
    plt.figure(figsize=(10, 6))
    # fit to exponential Gaussian
    _fit_y = gaussian_exp(trimmed_x, *_params)
    plt.plot(trimmed_x, _fit_y, lw=1.5, color='black', alpha=0.5, linestyle='--', label='Exponential Gaussian Fit')
    plt.plot(trimmed_x, trimmed_y, 'o', markersize=4, alpha=1, color=colors[2], label='Data')
    plt.title('Fitted Charge Pulse')
    # plot fit
    plt.xlabel('Time [µs]')
    plt.ylabel('Charge [ADC]')
    plt.legend()
    # plot data
    plt.grid(True)
    plt.show()
    print('Fitted Parameters:')
    print(f'  Amplitude: {_params[0]:.2f} ADC')
    print(f'  Mean (t_0): {_params[1]:.2f} µs')
    print(f'  Std Dev (σ): {_params[2]:.2f} µs')
    # print the fitted parameters
    print(f'  Tau (τ): {_params[3]:.2f} µs')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### uBOONe Response - 5th Order Semi-Gaussian
    """)
    return


@app.cell
def _(np):
    # uBOONe electronics response function; eq 2.1 from https://arxiv.org/pdf/1804.02583
    def ub_response(t, A0, tp, t0):
        # shift time axis by the offset t0 and ensure causality (response is 0 for t < t0)
        t_shifted = t - t0

        # amplitude coefficients
        A1 = 4.31054 * A0
        A2 = 2.6202 * A0
        A3 = 0.464924 * A0
        A4 = 0.762456 * A0
        A5 = 0.327684 * A0

        # exponential terms
        E1 = np.exp(-2.94809 * t_shifted / tp)
        E2 = np.exp(-2.82833 * t_shifted / tp)
        E3 = np.exp(-2.40318 * t_shifted / tp)

        # lambda parameters (terms inside sin/cos)
        L1 = 1.19361 * t_shifted / tp
        L2 = 2.38722 * t_shifted / tp
        L3 = 2.5928 * t_shifted / tp
        L4 = 5.18561 * t_shifted / tp

        # full response, from equation 2
        term1 = A1 * E1
        term2 = A2 * E2 * (np.cos(L1) + np.cos(L1) * np.cos(L2) + np.sin(L1) * np.sin(L2))
        term3 = A3 * E3 * (np.cos(L3) + np.cos(L3) * np.cos(L4) + np.sin(L3) * np.sin(L4))
        term4 = A4 * E2 * (np.sin(L1) - np.cos(L2) * np.sin(L1) + np.cos(L1) * np.sin(L2))
        term5 = A5 * E3 * (np.sin(L3) - np.cos(L4) * np.sin(L3) + np.cos(L3) * np.sin(L4))

        R = term1 - term2 + term3 + term4 - term5

        # response must be zero before the pulse starts
        R[t_shifted < 0] = 0

        return R

    return (ub_response,)


@app.cell
def _(
    colors,
    curve_fit,
    make_axes_locatable,
    plt,
    trimmed_x,
    trimmed_y,
    ub_response,
):
    A0_guess = 1.0
    tp_guess = 5.0
    t0_guess = 5.0
    p0_1 = [A0_guess, tp_guess, t0_guess]
    _params, _cov = curve_fit(ub_response, trimmed_x, trimmed_y, p0=p0_1, maxfev=5000)
    _fig, _ax1 = plt.subplots(figsize=(10, 6))
    _ax1.plot(trimmed_x, trimmed_y, 'o', color=colors[2], ms=4, label='Trimmed Data')
    _fit_y = ub_response(trimmed_x, *_params)
    _ax1.plot(trimmed_x, _fit_y, lw=2, color='black', alpha=0.5, linestyle='--', label='Fitted Electronics Response')
    _ax1.set_title('Fitted Charge Pulse')
    _ax1.set_ylabel('Charge [ADC]')
    _ax1.legend()
    _ax1.grid(True)
    divider = make_axes_locatable(_ax1)
    ax2 = divider.append_axes('bottom', size='30%', pad=0.1, sharex=_ax1)
    residuals = trimmed_y - _fit_y
    ax2.plot(trimmed_x, residuals, 'o', ms=2, color=colors[2], label='Residuals')
    ax2.axhline(0, color='r', linestyle='--', lw=1)
    ax2.set_ylabel('Residuals\n[ADC]')
    ax2.set_xlabel('Time [µs]')
    ax2.grid(True)
    plt.setp(_ax1.get_xticklabels(), visible=False)
    _fig.tight_layout(pad=1.0)
    plt.show()
    print('Fitted Parameters:')
    print(f'  A0 (Amplitude Scale): {_params[0]:.3f}')
    print(f'  tp (Peaking Time): {_params[1]:.3f} µs')
    print(f'  t0 (Time Offset): {_params[2]:.3f} µs')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3rd Order Semi-Gaussian
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Semi-analytically determined impulse response function for our pGRAMS shaper, which consists of 1 ideal differentiator and 1 lowpass integrator.
    """)
    return


@app.cell
def _(np):
    # 3rd order semi-gaussian response function; 1 ideal differentiator, 1 LP integrator
    def impulse_response(t, G, t0, tp, A0, A1, W1):
        # sigma and coefficient definitions
        sigma = tp * 1.0844
        # A0 = 1.2633573
        # A1 = 1.1490948
        # W1 = 0.7864188

        t_shifted = t - t0

        # common square term in both denominators
        square = (A0 - A1)**2 + W1**2

        # denominators
        D1 = square * sigma
        D2 = W1 * square * sigma

        # exponential terms
        E1 = np.exp(-A0 * t_shifted / sigma)
        E2 = np.exp(-A1 * t_shifted / sigma)

        # argument for sin/cos functions
        L = t_shifted * W1 / sigma

        # term 1
        term1 = E1/D1

        # term 2
        term2 = E2 * (W1 * np.cos(L) + (A1 - A0) * np.sin(L)) / D2

        # scaling factor for both terms
        scaling_factor = G * A0 * (A1**2 + W1**2)

        # response
        response = scaling_factor * (term1 - term2)

        # enforce causality: the response must be zero before the pulse starts at t0
        response[t_shifted < 0] = 0

        return response

    return (impulse_response,)


@app.cell
def _(colors, curve_fit, make_axes_locatable, np, plt, trapezoid):
    def fit_pulse(trimmed_x, trimmed_y, full_x_axis, full_waveform, integration_window, fit_function, p0, plot_title='Fitted Pulse', gaussian_func=None):
        """
        Performs a complete analysis for a given trimmed pulse:
        1. Fits the data with the provided function.
        2. Plots the fit result along with residuals.
        3. Plots a histogram of the residuals with a Gaussian fit.
        4. Calculates and displays a md table comparing the data and the fit.

        Args:
            trimmed_x (np.ndarray): The x-axis (time) of the trimmed pulse.
            trimmed_y (np.ndarray): The y-axis (charge) of the trimmed pulse.
            full_x_axis (np.ndarray): The x-axis (time) of the full waveform.
            full_waveform (np.ndarray): The full waveform data.
            integration_window (np.ndarray): The integration window for the pulse. First index = start time in us, second index = end time in us.
            fit_function (callable): The model function to use for fitting (e.g., impulse_response).
            p0 (list): A list of initial guesses for the fit parameters.
            plot_title (str): A title for the main fit plot.
            gaussian_func (callable, optional): A Gaussian function for the histogram fit.

        Returns:
            tuple: A tuple containing (fitted_parameters, residuals).
        """
        _params, _cov = curve_fit(fit_function, trimmed_x, trimmed_y, p0=p0, maxfev=5000)
        _fit_y = fit_function(trimmed_x, *_params)  # --- 1. Run the Fit ---
        full_fit = fit_function(full_x_axis, *_params)
        residuals = trimmed_y - _fit_y
        fig1, _ax1 = plt.subplots(figsize=(9, 6))
        _ax1.plot(trimmed_x, trimmed_y, 'o', color=colors[2], ms=4, label='Trimmed Data')
        _ax1.plot(trimmed_x, _fit_y, lw=1.2, color='grey', alpha=1, linestyle='--', label='Fitted Response')
        _ax1.set_title(plot_title)  # --- 2. Plot Fit and Residuals ---
        _ax1.set_ylabel('Charge [ADC]')
        _ax1.legend()
        _ax1.grid(True)
        divider = make_axes_locatable(_ax1)
        ax2 = divider.append_axes('bottom', size='30%', pad=0.1, sharex=_ax1)
        ax2.plot(trimmed_x, residuals, 'o', ms=2, color=colors[2])
        ax2.axhline(0, color='r', linestyle='--', lw=1)
        ax2.set_ylabel('Residuals\n[ADC]')
        ax2.set_xlabel('Time [µs]')
        ax2.grid(True)
        plt.setp(_ax1.get_xticklabels(), visible=False)
        fig1.tight_layout(pad=1.0)
        plt.show()
        if gaussian_func:
            plt.figure(figsize=(7, 4))
            bins = np.linspace(np.min(residuals) - 1, np.max(residuals) + 1, 15)
            hist, bin_edges = np.histogram(residuals, bins=bins, density=True)
            midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            try:  # --- 3. Plot Residuals Histogram ---
                p0_g = [np.mean(residuals), np.std(residuals)]
                params_g, _ = curve_fit(gaussian_func, midpoints, hist, p0=p0_g)
                mean_g, std_g = (params_g[0], params_g[1])
                x_g = np.linspace(bin_edges[0], bin_edges[-1], 100)
                plt.hist(residuals, bins=bins, histtype='step', color=colors[0], density=True)
                plt.fill_between(x_g, gaussian_func(x_g, *params_g), color=colors[9], alpha=0.5, label='Gaussian Fit')
                plt.axvline(mean_g, color='red', linestyle='--', lw=1, label='Mean')
                plt.text(0.05, 0.95, f'$\\mu$: {mean_g:.2f}\n$\\sigma$: {std_g:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
                plt.title(f'Residuals Histogram for {plot_title}')
                plt.xlabel('Residual ADC Counts')
                plt.legend()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f'Could not complete Gaussian fit for histogram: {e}')
        peak_amp_data = np.max(trimmed_y)
        peak_amp_fit = np.max(_fit_y)
        amp_rel_diff = (peak_amp_fit - peak_amp_data) / peak_amp_data * 100
        peak_time_data = trimmed_x[np.argmax(trimmed_y)]
        peak_time_fit = trimmed_x[np.argmax(_fit_y)]
        time_abs_diff = peak_time_fit - peak_time_data
        int_win = (2 * np.array(integration_window) + 255).astype(int)
        area_data = trapezoid(full_waveform[int_win[0]:int_win[1]], x=full_x_axis[int_win[0]:int_win[1]])
        area_fit = trapezoid(full_fit[int_win[0]:int_win[1]], x=full_x_axis[int_win[0]:int_win[1]])
        area_rel_diff = (area_fit - area_data) / area_data * 100  # --- 4. Display Comparison Table ---
        plt.figure(figsize=(10, 4))
        plt.plot(full_x_axis, full_fit, label='Full Fit', lw=1.2, color='grey', alpha=1, linestyle='--')
        plt.plot(full_x_axis, full_waveform, 'o', label='Data', color=colors[2], ms=2)
        plt.axvspan(full_x_axis[int_win[0]], full_x_axis[int_win[1]], color='#A0D3FF76', label='Integration Window')
        plt.title('Integration Window on ' + plot_title)
        plt.xlabel('Time [µs]')
        plt.ylabel('Charge [ADC]')
        plt.legend()  # always integrate over the same area from full waveform
        plt.grid(True)  # us input, converted to samples
        plt.show()
        markdown_table = f'\n| Metric | Data | Fit | Difference |\n|:---|:---:|:---:|:---:|\n| Peak Amplitude [ADC] | {peak_amp_data:.2f} | {peak_amp_fit:.2f} | {amp_rel_diff:+.2f}% |\n| Peak Time [µs] | {peak_time_data:.2f} | {peak_time_fit:.2f} | {time_abs_diff:+.3f} µs |\n| Integrated Area [ADC*µs] | {area_data:.2f} | {area_fit:.2f} | {area_rel_diff:+.2f}% |\n'
        return (_params, residuals, markdown_table)  # plot integration window on top of full fit

    return (fit_pulse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, using RMS based trimming:
    """)
    return


@app.cell
def _(
    fit_pulse,
    gaussian,
    impulse_response,
    inverted_waveform,
    mo,
    np,
    rms_1,
    trim_pulse,
    trimmed_x,
    trimmed_y,
    x_axis,
):
    # define the initial guesses once
    peak_height = np.max(trimmed_y)
    peak_time = trimmed_x[np.argmax(trimmed_y)]
    width_estimate = np.sum(trimmed_y > 0.5 * peak_height) * np.mean(np.diff(trimmed_x))

    p0_2 = [peak_height, peak_time, width_estimate, 1.0, 1.0, 1.0]

    # define integration window
    window = [-5, 10]
    # p0_2 = [1.0, 5.0, 5.0, 1.0, 1.0, 1.0]
    trimmed_x1, trimmed_y1 = trim_pulse(x_axis, inverted_waveform, rms_1, start_pad=-1, end_pad=0, plot=False)
    # --- Run 1: RMS-Trimmed ---
    params1, residuals1, table1 = fit_pulse(trimmed_x1, trimmed_y1, x_axis, inverted_waveform, window, fit_function=impulse_response, p0=p0_2, plot_title='RMS-Trimmed Fit', gaussian_func=gaussian)

    mo.md(table1)
    return p0_2, params1, window


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then, try ±10 around RMS based trimming:
    """)
    return


@app.cell
def _(
    fit_pulse,
    gaussian,
    impulse_response,
    inverted_waveform,
    mo,
    p0_2,
    rms_1,
    trim_pulse,
    window,
    x_axis,
):
    # --- Run 2: RMS±10-Trimmed ---
    trimmed_x2, trimmed_y2 = trim_pulse(x_axis, inverted_waveform, rms_1, start_pad=-10, end_pad=10, plot=False)
    params2, residuals2,table2 = fit_pulse(trimmed_x2, trimmed_y2, x_axis, inverted_waveform, window, impulse_response, p0_2, plot_title='RMS±10-Trimmed Fit', gaussian_func=gaussian)

    mo.md(table2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, with the undershoot contamination trimmed out.
    """)
    return


@app.cell
def _(
    fit_pulse,
    gaussian,
    impulse_response,
    inverted_waveform,
    mo,
    p0_2,
    rms_1,
    trim_pulse,
    window,
    x_axis,
):
    # --- Run 3: Undershoot-Trimmed ---
    trimmed_x3, trimmed_y3 = trim_pulse(x_axis, inverted_waveform, rms_1, start_pad=-1, end_pad=-11, plot=False)
    params3, residuals3, table3 = fit_pulse(trimmed_x3, trimmed_y3, x_axis, inverted_waveform, window, impulse_response, p0_2, plot_title='Undershoot-Trimmed Fit', gaussian_func=gaussian)
    mo.md(table3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The RMS-trimmed fit does the best job of balancing integrated area and peak amplitude both, so we use this for our electronics response. We list the best fit parameters to be implemented in GramsElecSim here.
    """)
    return


@app.cell
def _(params1):
    # print params with labels
    param_names = ['G', 't0', 'tp', 'A0', 'A1', 'W1']
    for i, p in enumerate(params1):
        print(f"Param {param_names[i]}: {p}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Measure Peak and Shaping Time
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
