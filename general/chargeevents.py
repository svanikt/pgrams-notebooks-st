import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""## November 2025 Integration Tests: Charge Events""")
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
    return curve_fit, find_peaks, mo, np, pd, plt


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
    mo.md(r"""## Load raw binary data""")
    return


@app.cell
def _(get_raw_data, np):

    num_files = 1
    run_number = '725'
    # run_number = '458' # ADC baselines

    files = []
    for i in np.arange(num_files):
        # files.append(f"/home/pgrams/data/nov2025_integration_data/readout_data/pGRAMS_bin_{run_number}_{i}.dat")
        files.append(f"/home/pgrams/data/jan13_integration/readout_data/pGRAMS_bin_{run_number}_{i}.dat")

    use_charge_roi = False
    readout_df = get_raw_data.get_event_data(files=files, light_slot=16, use_charge_roi=use_charge_roi, channel_threshold=[2055]*192)
    return (readout_df,)


@app.cell
def _(readout_df):
    readout_df.tail(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Preview Events""")
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

        baseline_subtract = 2048
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
        plt.xticks(np.arange(-int(timesize / 10) * 10, int((timesize * 2) / 10) * 10, 50))
        plt.title("Event " + str(event_df['event_index']))
        plt.xlabel("[$\\mu$s]")
        plt.ylabel("Charge [ADC]")
        if show_legend: plt.legend(loc='best')
        # if ch_down == ch_up or overlay: plt.ylim(range)
        if create_fig: plt.show()
    return (plot_charge_waveforms,)


@app.cell
def _(are_hits, find_hits, plot_charge_waveforms, plt, readout_df):
    """
    Plotting of both the charge and light signals.
    """
    light_channel = 0
    channel_range = [0,30]
    select_hit_events = True

    for event in range(0,4999):

        if are_hits(readout_df=readout_df, event=event):
            # modified to EXCLUDE events that have hits (including baseline drop out)
            # if are_hits(readout_df=readout_df, event=event) and select_hit_events:
            hit_channels = find_hits(readout_df, event, height=10, prom=2)
            if len(hit_channels)>0:
                _fig, ax = plt.subplots(figsize=(16,4))
                plot_charge_waveforms(event_df=readout_df.iloc[event], channels=hit_channels.astype(int), timesize=255, overlay=True, range=[1800, 2100], create_fig=False, show_legend=True)


                plt.xlabel("[$\\mu$s]")
                # plt.axvline(-128, color='red', linestyle='--')
                # plt.axvline(0, color='red', linestyle='--')
                plt.xlim(-130,260)
                plt.minorticks_on()
                plt.show()
        else:
            continue
    return


@app.cell
def _():
    neu_channels = ["B13", "A14", "B12", "A13", "B11", "A12", "B10", "A11", "B9", "A10", "B8", "A9", "B7", "A8", "B6", "A7", "B5", "A6", "B4", "A5", "B3", "A4", "B2", "A3", "B1", "A2", "B0", "A1", "GND", "A0"]

    return


@app.cell
def _(readout_df):
    x = [29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
    y = [26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0]


    event_idx = 4933
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
