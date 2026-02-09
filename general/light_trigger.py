import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    ## Integration Tests: Light Trigger
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
    colors = sns.color_palette('colorblind')

    # apply sexy Latex matplotlib style
    plt.style.use('/home/pgrams//latex-cm.mplstyle')


    # Add the location of the data utilities code to the system path
    # Can also just add it to PYTHONPATH but make sure you know where it is pointing
    # or you could end up using the wrong code.

    # The root directory for the utility code is up 2 directories from the notebook
    abs_repo_path = os.path.abspath('/home/pgrams/tpc_data/software/PGramsRawData')
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
    mo.md(r"""
    ## Load raw binary data
    """)
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
    num_files = 3
    # run_numbers = ['503','504','505']
    run_numbers = ['819']
    files = []
    for i in np.arange(len(run_numbers)):
        for j in np.arange(num_files):
            # files.append(f"/home/pgrams/data/nov2025_integration_data/pGRAMS_bin_{run_numbers[i]}_{j}.dat")
            files.append(f"/NAS/ColumbiaIntegration/readout_data/pGRAMS_bin_{run_numbers[i]}_{j}.dat")
    use_charge_roi = True
    readout_df = get_raw_data.get_event_data(files=files, light_slot=16, use_charge_roi=use_charge_roi, skip_beam_roi=True, channel_threshold=[2055]*192)

    # readout_df = readout_df.iloc[0:1049]
    return readout_df, run_numbers


@app.cell
def _():
    # readout_df.tail(10)
    return


@app.cell
def _(readout_df):
    len(readout_df)
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


    # plot.plot_light_waveforms(readout_df=readout_df, evt_range=(600,601), ylim=(1900,4096), show_diff=False, show_legend=False, show_events=False)
    return


@app.cell
def _(plot, readout_df):
    plot.plot_light_waveforms(readout_df=readout_df, evt_range=(1000,1002), ylim=(1900,4096), show_diff=False, show_legend=True, show_events=False)
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
    mo.md(r"""
    ## Plot Single-Channel Amplitude Distributions
    """)
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
            n, bins, patches = ax.hist(maxs, bins=binning, range=range, histtype='step', color=colors[2])

            total_counts = int(n.sum())
            ax.text(0.95, 0.95, f'N={total_counts}', transform=ax.transAxes, 
                    fontsize=7, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.axvline(2048+disc0, ls='--', lw='0.7', alpha=0.7, color='blue')
        ax.axvline(2048+disc1, ls='--', lw='0.7', alpha=0.7, color='red')
        ax.axvline(2048+trig, ls='--', lw='0.7', alpha=0.7, color='black')



        sipm_type = 'VUV'
        if sipm_channels[idx]%2==0:
            sipm_type = 'VIS'
        ax.set_title(f"Shaper Ch. {ch} ({sipm_type})", fontsize=9)
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
    plt.show()
    return binning, ch_maxs, range, runs, shaper_channels, sipm_channels


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
    mo.md(r"""
    ## Measure Trigger Rate
    """)
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
    # plt.xlim(0, 0.01)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



    print(params)
    return


@app.cell
def _(np, readout_df):
    trigger_frames = np.asarray(readout_df['trigger_frame_number'])
        # light_frames = np.asarray(readout_df['light_frame_number'].iloc[0])
        # light_channels = np.asarray(readout_df['light_channel'].iloc[0])
        # light_adc = np.asarray(readout_df['light_adc_words'].iloc[0])
    trigger_frames[2].size
    return


@app.cell
def _(np, plt, readout_df):
    # extract trigger pulse data and channels with filtering
    trigger_pulses = []
    trigger_channels_list = []

    for _ev in np.arange(len(readout_df)):
        _trig_frames = np.unique(readout_df['trigger_frame_number'][_ev])
        _light_frames = np.array(readout_df['light_frame_number'][_ev])
        _light_channels = np.array(readout_df['light_channel'][_ev])
        _light_adc = np.array(readout_df['light_adc_words'][_ev])

        for _tf in _trig_frames:
            _matches = np.where(_light_frames == _tf)[0]
            for _idx in _matches:
                _ch = int(_light_channels[_idx])
                if _ch <= 36:
                    # extract the pulse waveform
                    if _light_adc.ndim > 1:
                        _pulse = _light_adc[_idx]
                    else:
                        _pulse = _light_adc

                    # Filter: only keep pulses with length 256 and peak <= 4097
                    if len(_pulse) == 256 and np.max(_pulse) <= 4097:
                        trigger_pulses.append(_pulse)
                        trigger_channels_list.append(_ch)

    # Convert to arrays (now homogeneous since all are length 256)
    trigger_pulses = np.array(trigger_pulses)
    trigger_channels_list = np.array(trigger_channels_list)

    # Calculate peak amplitudes
    trigger_peak_amplitudes = np.max(trigger_pulses, axis=1) if len(trigger_pulses) > 0 else np.array([])

    # Plot distributions
    plt.figure(figsize=(8,5))

    # peak amplitude distribution
    plt.hist(trigger_peak_amplitudes, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Peak Amplitude (ADC)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Trigger Pulse Amplitude Distribution', 
                   fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # statistics
    print(f"Total trigger pulses: {len(trigger_pulses)}")
    print(f"Peak amplitude range: {trigger_peak_amplitudes.min():.1f} - {trigger_peak_amplitudes.max():.1f} ADC" if len(trigger_peak_amplitudes) > 0 else "No pulses")
    print(f"Mean peak amplitude: {trigger_peak_amplitudes.mean():.1f} ± {trigger_peak_amplitudes.std():.1f} ADC" if len(trigger_peak_amplitudes) > 0 else "No pulses")
    print(f"Median peak amplitude: {np.median(trigger_peak_amplitudes):.1f} ADC" if len(trigger_peak_amplitudes) > 0 else "No pulses")
    return (trigger_channels_list,)


@app.cell
def _(
    LogNorm,
    np,
    plt,
    readout_df,
    shaper_channels,
    sipm_channels,
    trigger_channels_list,
):
    # Count filtered triggers per channel
    filtered_trigger_counts = {}
    for _ch in trigger_channels_list:
        filtered_trigger_counts[_ch] = filtered_trigger_counts.get(_ch, 0) + 1

    # Build heatmap
    _heat = np.zeros((6, 6))
    for _i, _ch in enumerate(shaper_channels):
        _heat[_i//6, _i%6] = filtered_trigger_counts.get(_ch, 0)

    # Plot
    _fig, _ax = plt.subplots(figsize=(10, 10))

    if _heat.max() > 0:
        _im = _ax.imshow(_heat, cmap='hot', norm=LogNorm(vmin=1, vmax=_heat.max()))
    else:
        _im = _ax.imshow(_heat, cmap='hot')

    # Add colorbar
    _cbar = plt.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    _cbar.set_label('Number of Triggers', rotation=270, labelpad=15)

    # Annotations with larger, bolder texte
    for _i, _ch in enumerate(shaper_channels):
        _r, _c = _i//6, _i%6
        _val = int(_heat[_r, _c])
        _color = 'white' if _heat[_r, _c] < _heat.max()/2 else 'black'

        # Channel number and count
        _ax.text(_c, _r - 0.2, f'Ch{_ch}', 
                 ha='center', va='center', color=_color,
                 fontsize=12, fontweight='bold')

        _ax.text(_c, _r + 0.1, f'{_val}',
                 ha='center', va='center', color=_color,
                 fontsize=11, fontweight='bold')

        # SiPM type
        _sipm_type = 'VUV' if sipm_channels[_i] % 2 == 1 else 'VIS'
        _ax.text(_c, _r + 0.35, f'({_sipm_type})',
                 ha='center', va='center', color=_color,
                 fontsize=10, fontweight='bold')

    # Grid
    for _i in np.arange(7):
        _ax.axhline(_i-0.5, color='gray', lw=0.5, alpha=0.5)
        _ax.axvline(_i-0.5, color='gray', lw=0.5, alpha=0.5)

    # Clean up axes
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.set_xlim(-0.5, 5.5)
    _ax.set_ylim(5.5, -0.5)

    # Title
    _total_filtered = sum(filtered_trigger_counts.values())
    _ax.set_title(f'Trigger Heatmap {len(readout_df)} events)', 
                  fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
