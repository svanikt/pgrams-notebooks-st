import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    ## Notebook to test functions
    """)
    return


@app.cell
def _():
    import marimo as mo
    import sys 
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    # Apply the default matplotlib style to override marimo's dark theme
    plt.style.use('default')

    # Add the location of the data utilities code to the system path
    # Can also just add it to PYTHONPATH but make sure you know where it is pointing
    # or you could end up using the wrong code.

    # The root directory for the utility code is up 2 directories from the notebook
    abs_repo_path = os.path.abspath('/home/pgrams/tpc_data/software/PGramsRawData')
    # Insert the path to the front of sys.path if it is not already there
    if not abs_repo_path in sys.path:
        sys.path.insert(0, abs_repo_path)
    return mo, np, plt


@app.cell
def _():
    import raw_data_ana.get_data as get_raw_data
    import raw_data_ana.data_checks as data_checks
    import raw_data_ana.plotting as plot
    import raw_data_ana.charge_utils as qutils
    import raw_data_ana.light_utils as lutils
    return data_checks, get_raw_data, plot, qutils


@app.cell
def _(mo):
    mo.md(r"""
    ## Load raw binary data
    22-25
    """)
    return


@app.cell
def _(get_raw_data):
    # files = "/home/pgrams/data/readout_data/pGRAMS_bin_156_0.dat"
    # files = "/home/pgrams/data/sabertooth_data/pGRAMS_bin_123_15.dat" # Slow 900Hz --> corrupt
    # files = "/home/pgrams/data/sabertooth_data/pGRAMS_bin_127_0.dat" # Fast 900Hz --> corrupt
    # files = "/home/pgrams/data/sabertooth_data/pGRAMS_bin_130_0.dat" # Fast 800Hz --> okay 
    # files = ["/home/pgrams/data/sabertooth_data/pGRAMS_bin_131_0.dat", # Slow 800Hz --> okay
    #         "/home/pgrams/data/sabertooth_data/pGRAMS_bin_131_1.dat" ]
    # files = ["/home/pgrams/data/sabertooth_data/pGRAMS_bin_132_0.dat", # Slow 850Hz --> okay
    #         "/home/pgrams/data/sabertooth_data/pGRAMS_bin_132_1.dat" ]
    # files = "/home/pgrams/data/sabertooth_data/pGRAMS_bin_137_0.dat" # Fast 850Hz --> okay
    # files = "/home/pgrams/data/sabertooth_data/pGRAMS_bin_316_53.dat" #
    files = "/NAS/ColumbiaIntegration/readout_data/pGRAMS_bin_818_0.dat" #
    use_charge_roi = False
    readout_df = get_raw_data.get_event_data(files=files, light_slot=16, use_charge_roi=use_charge_roi, channel_threshold=[2055]*192)
    return (readout_df,)


@app.cell
def _():
    # readout_df.tail(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Perform basic data quality checks
    """)
    return


@app.cell
def _(data_checks, readout_df):
    # data_checks.check_fems(readout_df=readout_df)
    data_checks.word_count_check(readout_df=readout_df)
    return


@app.cell
def _(plot, plt, readout_df):
    plot.plot_trigger_rate(readout_df=readout_df, fem_num=0, bin_range=[200,1200])
    plt.show()
    return


@app.cell
def _(plot, readout_df):
    evt = 102
    plot.plot_charge_channels_df(event_df=readout_df.iloc[evt], num_channel=192, timesize=255, charge_range=(2000,2100))
    return


@app.cell
def _(plot, readout_df):
    # for event in range(len(readout_df)):
    for event in range(653,657):
        if len(readout_df['charge_adc_idx'][event]) > 0:
            plot.plot_charge_and_light(event_df=readout_df.iloc[event], light_channel=0, charge_range=[0,30], 
                                       x_range=[-150,260], y_range=[2000,2100], show_legend=True)
    return


@app.cell
def _(plot, readout_df):
    plot.plot_light_waveforms(readout_df=readout_df, evt_range=(5,9), ylim=(1900,2800), show_diff=False, show_legend=True, show_events=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Channel Mapping & Event Display
    """)
    return


@app.cell
def _(plot, qutils, readout_df):
    evt_map = 654
    if len(readout_df['charge_adc_idx'].iloc[evt_map]) > 0: 
        channel_words = qutils.full_charge_waveform(event_df=readout_df.iloc[evt_map], num_channel=30, timesize=255)
    else:
        channel_words = readout_df['charge_adc_words'][evt_map][:30,:]

    bottom,top = qutils.remap_channels(channel_arr=channel_words, detector='ugrams')

    print("Bottom Channels")
    plot.plot_charge_channels(channel_adc=bottom, event=evt_map, num_channel=len(bottom), timesize=255, charge_range=(2000,2100))
    print("Top Channels")
    plot.plot_charge_channels(channel_adc=top, event=evt_map, num_channel=len(top), timesize=255, charge_range=(2000,2100))
    return bottom, top


@app.cell
def _(bottom, plot, top):
    plot.ugrams_event_display(xchannels=bottom, ychannels=top, charge_thresh=2056)
    return


@app.cell
def _(plt, readout_df):
    plt.figure(figsize=(12,5))
    plt.hist(readout_df['charge_adc_words'][654][0,:], range=(1024,4096), bins=16)
    plt.show()
    return


@app.cell
def _(np):
    def rolling_avg(adc, avg_window):
        cumsum_vec = np.cumsum(np.insert(adc, 0, 0)) 
        return (cumsum_vec[avg_window:] - cumsum_vec[:-avg_window]) / avg_window

    def absrs(adc, n):
        yn = []
        yn1 = 0
        for word in adc:
            yn.append((1./n)*word + (1. - (1./n))*yn1)
            yn1 = yn[-1]
        return np.asarray(yn)
    return absrs, rolling_avg


@app.cell
def _(absrs, plt, readout_df, rolling_avg):
    ch_wf = readout_df['charge_adc_words'][79][24,:]

    plt.figure(figsize=(15,5))
    plt.plot(ch_wf, label="Waveform")
    n = 1./20.
    abst = absrs(adc=ch_wf,n=10)
    plt.plot(abst[8:], label="Running Sum (n=10)")
    # plt.plot(0.998*(n*ch_wf[1:] + (1-n) * ch_wf[:-1]))
    ra = rolling_avg(adc=ch_wf, avg_window=10)
    plt.plot(ra, label="Rolling Avg (n=10)")
    # readout_df['charge_channel'][79]
    plt.ylim(2034,2065)
    plt.legend()
    plt.show()
    return


@app.cell
def _(readout_df):
    all_rois_min = []
    all_rois_max = []
    all_roi_ch = []
    for i, (allch, lw) in enumerate(zip(readout_df["light_channel"], readout_df["light_adc_words"])):
        if len(allch) < 1 or len(lw) < 1:
            continue
        for rch, rmin, rmax in zip(allch, lw.min(axis=1), lw.max(axis=1)):
            all_rois_min.append(rmin)
            all_rois_max.append(rmax)
            all_roi_ch.append(rch)
            if rmax < 2500:
                print(i, "/", rch)
    return all_roi_ch, all_rois_max, all_rois_min


@app.cell
def _(all_rois_max, all_rois_min, plt):
    bins = 200
    xrange = (2000, 2800)

    plt.figure(figsize=(12,5))
    plt.hist(all_rois_min, bins=bins, range=xrange)
    plt.hist(all_rois_max, bins=bins, range=xrange)
    plt.xlabel("ADC")
    plt.yscale('log')
    plt.grid()
    plt.show()
    return bins, xrange


@app.cell
def _(all_roi_ch, all_rois_max, all_rois_min, bins, plt, xrange):
    from matplotlib.colors import LogNorm

    plt.figure(figsize=(12,5))
    plt.hist2d(all_rois_min, all_roi_ch, bins=[bins, 36], norm=LogNorm(), range=[xrange, [0,36]], cmin=1)
    plt.colorbar()
    plt.hist2d(all_rois_max, all_roi_ch, bins=[bins, 36], range=[xrange, [0,36]], cmin=1)
    plt.grid()
    plt.xlabel("ADC")
    plt.ylabel("Channel")
    plt.show()
    return


@app.cell
def _():
    # readout_df[50:100]
    return


@app.cell
def _(readout_df):
    len_lch_list = []
    for lch in readout_df["light_channel"]:
        len_lch_list.append(len(lch))
    return (len_lch_list,)


@app.cell
def _(len_lch_list, plt):
    plt.hist(len_lch_list[1:], bins=36, range=[0,36])
    plt.grid()
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
