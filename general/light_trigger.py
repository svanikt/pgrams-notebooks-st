import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""## Integration Tests: Light Trigger""")
    return


@app.cell
def _():
    import marimo as mo
    import sys 
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    colors = sns.color_palette('colorblind')

    # apply sexy Latex matplotlib style
    plt.style.use('/home/pgrams//latex-cm.mplstyle')


    # Add the location of the data utilities code to the system path
    # Can also just add it to PYTHONPATH but make sure you know where it is pointing
    # or you could end up using the wrong code.

    # The root directory for the utility code is up 2 directories from the notebook
    abs_repo_path = os.path.abspath('../../')
    # Insert the path to the front of sys.path if it is not already there
    if not abs_repo_path in sys.path:
        sys.path.insert(0, abs_repo_path)
    return colors, mo, np, plt


@app.cell
def _():
    import raw_data_ana.get_data as get_raw_data
    import raw_data_ana.data_checks as data_checks
    import raw_data_ana.plotting as plot
    import raw_data_ana.charge_utils as qutils
    import raw_data_ana.light_utils as lutils
    return data_checks, get_raw_data, plot


@app.cell
def _(mo):
    mo.md(r"""## Load raw binary data""")
    return


@app.cell
def _():
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
    # use_charge_roi = False
    # readout_df = get_raw_data.get_event_data(files=files, light_slot=16, use_charge_roi=use_charge_roi, channel_threshold=[2055]*192)
    return


@app.cell
def _(get_raw_data, np):
    # /NAS/ColumbiaIntegration/
    num_files = 2
    run_numbers = ['503','504','505']

    files = []
    for i in np.arange(len(run_numbers)):
        for j in np.arange(num_files):
            files.append(f"/home/pgrams/data/nov2025_integration_data/pGRAMS_bin_{run_numbers[i]}_{j}.dat")

    use_charge_roi = True
    readout_df = get_raw_data.get_event_data(files=files, light_slot=16, use_charge_roi=use_charge_roi, channel_threshold=[2055]*192)

    # readout_df = readout_df.iloc[0:1049]
    return readout_df, run_numbers


@app.cell
def _(readout_df):
    readout_df.tail(10)
    return


@app.cell
def _(readout_df):
    len(readout_df)
    return


@app.cell
def _(mo):
    mo.md(r"""## Perform basic data quality checks""")
    return


@app.cell
def _(data_checks, readout_df):
    # data_checks.check_fems(readout_df=readout_df)
    data_checks.word_count_check(readout_df=readout_df)
    return


@app.cell
def _():
    evt = 102
    # plot.plot_charge_channels_df(event_df=readout_df.iloc[evt], num_channel=192, timesize=255, charge_range=(2000,2100))
    return


@app.cell
def _():
    # # for event in range(len(readout_df)):
    # for event in range(653,657):
    #     if len(readout_df['charge_adc_idx'][event]) > 0:
    #         plot.plot_charge_and_light(event_df=readout_df.iloc[event], light_channel=0, charge_range=[0,30], 
    #                                    x_range=[-150,260], y_range=[2000,2100], show_legend=True)
    return


@app.cell
def _():
    # plot.plot_light_waveforms(readout_df=readout_df, evt_range=(4640,4700), ylim=(1900,4096), show_diff=False, show_legend=False, show_events=False)
    return


@app.cell
def _():
    # plot.plot_light_waveforms(readout_df=readout_df, evt_range=(4643,4644), ylim=(1900,4096), show_diff=False, show_legend=False, show_events=False)
    return


@app.cell
def _():
    # all_rois_min = []
    # all_rois_max = []
    # all_roi_ch = []
    # for i, (allch, lw) in enumerate(zip(readout_df["light_channel"], readout_df["light_adc_words"])):
    #     if len(allch) < 1 or len(lw) < 1:
    #         continue
    #     for rch, rmin, rmax in zip(allch, lw.min(axis=1), lw.max(axis=1)):
    #         all_rois_min.append(rmin)
    #         all_rois_max.append(rmax)
    #         all_roi_ch.append(rch)
    #         if rmax < 2500:
    #             print(i, "/", rch)
    return


@app.cell
def _():
    # bins = 200
    # xrange = (2000, 2800)

    # plt.figure(figsize=(12,5))
    # plt.hist(all_rois_min, bins=bins, range=xrange)
    # plt.hist(all_rois_max, bins=bins, range=xrange)
    # plt.xlabel("ADC")
    # plt.yscale('log')
    # plt.grid()
    # plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Plot Single-Channel Amplitude Distributions""")
    return


@app.cell
def _(colors, plt, readout_df, run_numbers):
    from collections import defaultdict
    from matplotlib.lines import Line2D

    # collect per-channel amplitude lists
    ch_mins = defaultdict(list)
    ch_maxs = defaultdict(list)

    for channels, words in zip(readout_df["light_channel"], readout_df["light_adc_words"]):
        if len(channels) < 1 or len(words) < 1:
            continue
        # min/max amplitudes for each ROI
        try: 
            rmins = words.min(axis=1) 
        except ValueError:
            pass
        try: 
            rmaxs = words.max(axis=1) 
        except ValueError:
            pass
        for ch, min, max in zip(channels, rmins, rmaxs):
            if ch > 36:
                # print(ch)
                continue
            ch_mins[ch].append(min)
            ch_maxs[ch].append(max)

    # ensure we have exactly 36 channels to plot
    shaper_channels = [0,2,4,6,7,9,1,3,5,32,8,10,18,20,14,16,11,13,19,34,15,17,12,33,21,23,25,27,28,30,22,24,26,35,29,31]
    # shaper_channels = [0,2,4,6,7,9,1,3,5,32,8,10,18,20,14,16,11,13,19,34,15,17,12,33,21,23,25,27,28,30,22,24,36,35,29,31]

    sipm_channels = [1,3,5,7,9,11,2,4,6,8,10,12,1,3,5,7,9,11,2,4,6,8,10,12,1,3,5,7,9,11,2,4,6,8,10,12]


    # hist settings
    binning = 200
    range = (2048, 4096)

    # prepare figure
    nrows, ncols = 6, 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    disc0=150
    disc1=160
    trig = disc1*2

    for idx, ch in enumerate(shaper_channels):
        ax = axes[idx]
        mins = ch_mins.get(ch, [])
        maxs = ch_maxs.get(ch, [])

        # plot only if there is data
        # if len(mins) > 0:
            # ax.hist(mins, bins=binning, range=range, histtype='step', color='darkblue', label='ADC min')
        if len(maxs) > 0:
            ax.hist(maxs, bins=binning, range=range, histtype='step', color=colors[2])

        ax.axvline(2048+disc0, ls='--', lw='0.7', alpha=0.7, color='blue')
        ax.axvline(2048+disc1, ls='--', lw='0.7', alpha=0.7, color='red')
        ax.axvline(2048+trig, ls='--', lw='0.7', alpha=0.7, color='black')


        sipm_type = 'VUV'
        if sipm_channels[idx]%2==0:
            sipm_type = 'VIS'
        ax.set_title(f"Shaper Channel {ch} ({sipm_type})", fontsize=9)
        ax.set_yscale('log')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        # show legend only if both curves exist
        # if len(mins) > 0 and len(maxs) > 0:
        #     ax.legend(fontsize=8, frameon=False)

    # global labels
    fig.text(0.5, 0.04, "ADC", ha='center')
    fig.text(0.06, 0.5, "Counts", va='center', rotation='vertical')
    if len(run_numbers)>1:
        runs = ", ".join(str(run) for run in run_numbers)
        fig.suptitle(f'ADC Amplitude Distributions for Runs {runs} ({len(readout_df)} events)')
    else:
        fig.suptitle(f'ADC Amplitude Distributions for Run {run_numbers[0]} ({len(readout_df)} events)')
    disc0_label = Line2D([0], [0], color='blue', linestyle='--', alpha=0.7, label=f'disc0={disc0}')
    disc1_label = Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label=f'disc1={disc1}')
    trig_label = Line2D([0], [0], color='black', linestyle='--', alpha=0.7, label=f'summed ADC threshold={trig}')
    fig.legend(handles=[disc0_label, disc1_label, trig_label], loc='lower center', ncols=3, frameon=False, bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.97])

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.97])
    plt.show()
    return binning, ch_maxs, range, runs


@app.cell
def _(binning, ch_maxs, colors, plt, range):
    # look at specific channel distributions
    plt.figure(figsize=[8,6])
    plt.hist(ch_maxs[14],bins=binning, range=range, histtype='step', color='teal', label='14')
    plt.hist(ch_maxs[15],bins=binning, range=range, histtype='step', color=colors[3], label='15')
    plt.hist(ch_maxs[16],bins=binning, range=range, histtype='step', color='red', label='16')
    plt.hist(ch_maxs[17],bins=binning, range=range, histtype='step', color=colors[2], label='17')
    plt.xlim(2000,4096)
    plt.semilogy()
    plt.title("Thorium Cell Channels")
    plt.legend()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""## Measure Light Trigger Rate""")
    return


@app.cell
def _():
    # trig_rate[12825]
    # readout_df[12824:12829]
    # np.arange(len(intervals))[intervals<0]
    return


@app.cell
def _(colors, np, plot, plt, readout_df):
    trig_rate = plot.plot_trigger_rate(readout_df=readout_df, fem_num=0, bin_range=[0,600], timesize=255)


    intervals = 1/np.asarray(trig_rate)
    intervals = intervals[intervals>0]
    num_bins=100


    plt.figure(figsize=(8,5))
    counts, edges, _ = plt.hist(intervals, bins=num_bins, density=True, alpha=0.6, color=colors[2])

    plt.xlabel(r'$\Delta$t (s)')
    plt.ylabel("Density")
    # plt.legend()
    plt.tight_layout()
    plt.show()
    return counts, edges, intervals, num_bins


@app.cell
def _(counts, edges, np):
    from scipy.optimize import curve_fit
    def exponential_fit(x, lam):
        return lam*np.exp(-lam * x)

    bin_centers = 0.5 * (edges[1:] + edges[:-1])

    params, covariance = curve_fit(exponential_fit, bin_centers[0:-1], counts[0:-1], p0=[1])

    fitted_curve = exponential_fit(bin_centers, *params)
    return bin_centers, fitted_curve, params


@app.cell
def _(
    bin_centers,
    colors,
    fitted_curve,
    intervals,
    num_bins,
    params,
    plt,
    readout_df,
    run_numbers,
    runs,
):
    # plot histogram and fitted curve
    plt.figure(figsize=(8, 6))
    plt.hist(intervals, bins=num_bins, density=True, histtype='bar', alpha=0.4, color=colors[2])
    plt.plot(bin_centers, fitted_curve, color=colors[2], label='Exp. Fit', lw=1.5)
    plt.axvline(1/params[0], color='red',lw=1, ls='--', label=f'1/$\\lambda$ = {1/params[0]:.3f} s')

    plt.annotate(xy=[0.5,2.5], color='red', text=f'Fitted $\\lambda$ = {params[0]:.3f} Hz', fontsize=16)

    plt.xlabel(r'$\Delta$t (s)')
    plt.ylabel('Counts')

    if len(run_numbers)>1:
        plt.title(f'Light Trigger Rate Fit for Runs {runs} ({len(readout_df)} events)')
    else:
        plt.title(f'Light Trigger Rate Fit for Run {run_numbers[0]} ({len(readout_df)} events)')
    plt.xlim(0, 0.01)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



    print(params)
    return


@app.cell
def _():
    # evt_map = 654
    # if len(readout_df['charge_adc_idx'].iloc[evt_map]) > 0: 
    #     channel_words = qutils.full_charge_waveform(event_df=readout_df.iloc[evt_map], num_channel=30, timesize=255)
    # else:
    #     channel_words = readout_df['charge_adc_words'][evt_map][:30,:]

    # bottom,top = qutils.remap_channels(channel_arr=channel_words, detector='ugrams')

    # print("Bottom Channels")
    # plot.plot_charge_channels(channel_adc=bottom, event=evt_map, num_channel=len(bottom), timesize=255, charge_range=(2000,2100))
    # print("Top Channels")
    # plot.plot_charge_channels(channel_adc=top, event=evt_map, num_channel=len(top), timesize=255, charge_range=(2000,2100))
    return


@app.cell
def _():
    # plot.ugrams_event_display(xchannels=bottom, ychannels=top, charge_thresh=2056)
    return


@app.cell
def _():
    # plt.figure(figsize=(12,5))
    # plt.hist(readout_df['charge_adc_words'][654][0,:], range=(1024,4096), bins=16)
    # plt.show()
    return


@app.cell
def _():
    # def rolling_avg(adc, avg_window):
    #     cumsum_vec = np.cumsum(np.insert(adc, 0, 0)) 
    #     return (cumsum_vec[avg_window:] - cumsum_vec[:-avg_window]) / avg_window

    # def absrs(adc, n):
    #     yn = []
    #     yn1 = 0
    #     for word in adc:
    #         yn.append((1./n)*word + (1. - (1./n))*yn1)
    #         yn1 = yn[-1]
    #     return np.asarray(yn)
    return


@app.cell
def _():
    # ch_wf = readout_df['charge_adc_words'][79][24,:]

    # plt.figure(figsize=(15,5))
    # plt.plot(ch_wf, label="Waveform")
    # n = 1./20.
    # abst = absrs(adc=ch_wf,n=10)
    # plt.plot(abst[8:], label="Running Sum (n=10)")
    # # plt.plot(0.998*(n*ch_wf[1:] + (1-n) * ch_wf[:-1]))
    # ra = rolling_avg(adc=ch_wf, avg_window=10)
    # plt.plot(ra, label="Rolling Avg (n=10)")
    # # readout_df['charge_channel'][79]
    # plt.ylim(2034,2065)
    # plt.legend()
    # plt.show()
    return


@app.cell
def _():
    # all_rois_min = []
    # all_rois_max = []
    # all_roi_ch = []
    # for i, (allch, lw) in enumerate(zip(readout_df["light_channel"], readout_df["light_adc_words"])):
    #     if len(allch) < 1 or len(lw) < 1:
    #         continue
    #     for rch, rmin, rmax in zip(allch, lw.min(axis=1), lw.max(axis=1)):
    #         all_rois_min.append(rmin)
    #         all_rois_max.append(rmax)
    #         all_roi_ch.append(rch)
    #         if rmax < 2500:
    #             print(i, "/", rch)
    return


@app.cell
def _():
    # bins = 200
    # xrange = (2000, 2800)

    # plt.figure(figsize=(12,5))
    # plt.hist(all_rois_min, bins=bins, range=xrange)
    # plt.hist(all_rois_max, bins=bins, range=xrange)
    # plt.xlabel("ADC")
    # plt.yscale('log')
    # plt.grid()
    # plt.show()
    return


@app.cell
def _():
    # plt.figure(figsize=(12,5))
    # plt.hist2d(all_rois_min, all_roi_ch, bins=[bins, 15], range=[xrange, [0,15]], cmin=1)
    # plt.colorbar()
    # plt.hist2d(all_rois_max, all_roi_ch, bins=[bins, 15], range=[xrange, [0,15]], cmin=1)
    # plt.grid()
    # plt.xlabel("ADC")
    # plt.ylabel("Channel")
    # plt.show()
    return


@app.cell
def _(readout_df):
    readout_df[50:100]
    return


@app.cell
def _(readout_df):
    len_lch_list = []
    for lch in readout_df["light_channel"]:
        len_lch_list.append(len(lch))
    return (len_lch_list,)


@app.cell
def _(len_lch_list, plt):
    plt.hist(len_lch_list[1:], bins=10, range=[0,10])
    plt.grid()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
