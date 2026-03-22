import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    ## Charge Baselines
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
    colors = sns.color_palette('colorblind')
    # sns.set_theme()


    # apply sexy Latex matplotlib style
    plt.style.use('default')
    plt.style.use('/home/pgrams/latex-cm.mplstyle')


    # Add the location of the data utilities code to the system path
    # Can also just add it to PYTHONPATH but make sure you know where it is pointing
    # or you could end up using the wrong code.

    # abs_repo_path = os.path.abspath('/home/pgrams/tpc_data/software/PGramsRawData')
    abs_repo_path = os.path.abspath('/home/pgrams/daq_analysis/PGramsRawData')
    # Insert the path to the front of sys.path if it is not already there
    if not abs_repo_path in sys.path:
        sys.path.insert(0, abs_repo_path)
    return colors, curve_fit, mo, np, pd, plt


@app.cell
def _():
    import raw_data_ana.get_data as get_raw_data
    import raw_data_ana.data_checks as data_checks
    import raw_data_ana.plotting as plot
    import raw_data_ana.charge_utils as qutils
    import raw_data_ana.light_utils as lutils
    import raw_data_ana.hit_finding as hf

    return get_raw_data, hf, plot


@app.cell
def _(mo):
    mo.md(r"""
    ## Load raw binary data
    """)
    return


@app.cell
def _(get_raw_data, np):
    num_files = 1
    run_number = '774'

    files = []
    for i in np.arange(num_files):
        # files.append(f"/NAS/ColumbiaIntegration/readout_data/pGRAMS_bin_{run_number}_{i}.dat")
        files.append(f"/home/pgrams/data/sabertooth2_data/data/readout_data/pGRAMS_bin_{run_number}_{i}.dat")
    use_charge_roi = False
    readout_df = get_raw_data.get_event_data(files=files, light_slot=16, use_charge_roi=False, channel_threshold=[2055]*192)
    return readout_df, run_number


@app.cell
def _(readout_df):
    readout_df.tail(1)
    return


@app.cell
def _(hf, readout_df):
    # Reload the module to pick up any changes
    import importlib
    importlib.reload(hf)

    # Test again
    for _evt in range(2192, 2196):
        _result = hf.are_hits(readout_df=readout_df, event=_evt)
        print(f"Event {_evt}: are_hits = {_result}")
    return


@app.cell
def _(np, readout_df):
    # Check what the function is actually seeing
    _evt = 2192
    _charge_via_function = readout_df['charge_adc_words'][_evt]
    _charge_direct = readout_df.iloc[_evt]['charge_adc_words']

    print(f"Via indexing: shape = {_charge_via_function.shape}")
    print(f"Via iloc: shape = {_charge_direct.shape}")
    print(f"Are they the same? {np.array_equal(_charge_via_function, _charge_direct)}")
    return


@app.cell
def _(np, readout_df):
    # Manually replicate are_hits logic
    _evt = 2192
    _charge = readout_df['charge_adc_words'][_evt]
    print(f"Shape: {_charge.shape}")

    _found_jump = False
    for _ch in range(_charge.shape[0]):
        _max_diff = np.max(np.abs(np.diff(_charge[_ch, :])))
        if _max_diff > 8:
            print(f"Channel {_ch}: max diff = {_max_diff}")
            _found_jump = True

    if not _found_jump:
        print("No jumps > 8 found")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preview Events
    """)
    return


@app.cell
def _(hf, plot, plt, readout_df):
    # preview a few of the baseline waveforms
    light_channel = 0
    channel_range = [0,192]
    select_hit_events = True

    for event in range(2192,2195):

        if not hf.are_hits(readout_df=readout_df, event=event):
            # modified to EXCLUDE events that have hits (including baseline drop out)
            # if are_hits(readout_df=readout_df, event=event) and select_hit_events:
            # hit_channels = hf.find_hits(readout_df, event, height=1, prom=1)
            # if len(hit_channels)>0:
            fig, ax1 = plt.subplots(figsize=(16,4))
            plot.plot_charge_waveforms(event_df=readout_df.iloc[event], channel_range=[channel_range[0], channel_range[1]], timesize=255, overlay=True, range=[420, 520], create_fig=False, show_legend=False)

            plt.xlabel("[$\\mu$s]")
            plt.axvline(-128, color='red', linestyle='--')
            plt.axvline(0, color='red', linestyle='--')
            plt.axhline(0.61*(-15), color='red', linestyle='--')
            plt.xlim(-128,156)
            plt.minorticks_on()
            from matplotlib.ticker import AutoMinorLocator
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.yaxis.set_minor_locator(AutoMinorLocator())
            ax1.grid(True, which='major', linestyle='-', linewidth=0.8)
            ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)


            plt.show()
        else:
            continue
    return channel_range, select_hit_events


@app.cell
def _(channel_range):
    counts = {ch: [] for ch in range(channel_range[0], channel_range[1])}
    return (counts,)


@app.cell
def _(channel_range, counts, hf, np, readout_df, select_hit_events):
    for evt in range(0, len(readout_df) - 2):
        if readout_df['charge_adc_words'][evt].ndim < 2:
            continue

        num_channels = readout_df['charge_adc_words'][evt].shape[0]
        if num_channels < channel_range[1]: 
            print(f'event {evt} has only {num_channels} channels...') 
            continue

        if hf.are_hits(readout_df=readout_df, event=evt) and select_hit_events:
            continue

        for ch in np.arange(channel_range[0], channel_range[1]):
            counts[ch].append(readout_df['charge_adc_words'][evt][ch, 0:125])
    return


@app.cell
def _(np):
    def gaussian(x, mu, sigma):
        return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)

    return (gaussian,)


@app.cell
def _(colors, curve_fit, gaussian, np, pd, plt):
    def noise_hist(counts, channels=np.arange(0,30), baseline=2048, range=20, num_bins=40, subplot=10, saveas=''):
        bins = np.linspace(baseline - (range/2), baseline + (range/2), num_bins)
        baselines = {} 
        rms = {}

        if len(channels)==1: subplot=0

        if subplot:
            rows = len(channels) // subplot + (len(channels) % subplot > 0)
            if len(channels)>=subplot: 
                _, axs = plt.subplots(rows, subplot, figsize=(20, 2*rows))
            else: 
                _, axs = plt.subplots(rows, len(channels), figsize=(20, 2*len(channels)))

        for i, ch in enumerate(channels):
            if subplot:
                row = i//subplot
                col = i%subplot
                ax = axs[row, col]
            else:
                plt.figure(figsize=(7, 6))
                ax = plt.gca()

            try:
                if len(counts[ch]) == 0:
                    raise ValueError("Empty channel")

                all_counts = np.concatenate([np.ravel(arr) for arr in counts[ch]]).astype(float)
                all_counts = all_counts[np.isfinite(all_counts)]

                if len(all_counts) < 2 or np.std(all_counts) == 0:
                    raise ValueError("Insufficient valid data")

                mean = np.mean(all_counts)
                std = np.std(all_counts)

                hist, bin_edges = np.histogram(all_counts, bins=bins, density=True)
                midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])

                params, _ = curve_fit(gaussian, midpoints, hist, [mean, std])
                mean_fit, std_fit = params

                baselines[ch] = mean_fit
                rms[ch] = std

                ax.hist(all_counts, bins=bins, histtype='step', color=colors[0], density=True)

                x = np.linspace(min(bins), max(bins), range)
                fit = gaussian(x, mean, std)
                ax.fill_between(x, 0, fit, color=colors[9], alpha=0.5, label='Gaussian Fit')

                ax.axvline(baseline, color='r', ls='--', alpha=0.5, label='Expected BL')   
                ax.axvline(mean, color='r', ls='--', label='Actual BL')

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

            except Exception as e:
                baselines[ch] = np.nan
                rms[ch] = np.nan
                ax.set_title(f'Ch. {ch} - {str(e)[:20]}')
                ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, ha='center', va='center')


        plt.tight_layout()

        results = {
            "Ch": list(baselines.keys()),
            "Mean Counts": list(baselines.values()),
            "RMS Counts": list(rms.values())
        }
        results_df = pd.DataFrame(results)

        return baselines, rms


    return (noise_hist,)


@app.cell
def _(counts, noise_hist, np, plt):
    baselines, rms= noise_hist(counts, channels=np.arange(0,192), baseline=470, range=40, num_bins=40, subplot=6)
    plt.show()
    return baselines, rms


@app.cell
def _(baselines, colors, plt):
    channels = list(baselines.keys())
    baseline = [baselines[ch] for ch in channels]
    # neu_channels = ["B13", "A14", "B12", "A13", "B11", "A12", "B10", "A11", "B9", "A10", "B8", "A9", "B7", "A8", "B6", "A7", "B5", "A6", "B4", "A5", "B3", "A4", "B2", "A3", "B1", "A2", "B0", "A1", "GND", "A0"]

    figure, ax = plt.subplots(figsize=(10, 5))
    ax.bar(channels, baseline, color=colors[9], alpha=0.4)
    ax.set_xlabel('CU ADC Channel', fontsize=14)
    ax.set_ylabel('Baseline (ADC Counts)', fontsize=14)
    # ax.set_xticks(channels)
    # ax.set_xticklabels([str(ch) for ch in channels], rotation=90, fontsize=4)

    # Add secondary x-axis
    # ax3 = ax.twiny()
    # ax3.set_xlim(ax.get_xlim())
    # ax3.set_xticks(channels)
    # ax3.set_xticklabels([str(ch) for ch in neu_channels], rotation=90, fontsize=10)
    # ax3.set_xlabel('NEU TPC Channel', fontsize=14)

    # plt.ylim(2030, 2060)
    plt.ylim(440, 500)
    plt.tight_layout()
    plt.show()
    return baseline, channels


@app.cell
def _(channels, colors, plt, rms):
    noise_rms = [rms[ch] for ch in channels]

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.bar(channels, noise_rms, color=colors[2], alpha=0.4)
    _ax.set_xlabel('ADC Channel', fontsize=14)
    _ax.set_ylabel('Noise RMS (ADC Counts)', fontsize=14)
    # _ax.set_xticks(channels)
    # _ax.set_xticklabels([str(ch) for ch in channels], rotation=90, fontsize=10)

    # Add secondary x-axis
    # _ax2 = _ax.twiny()
    # _ax2.set_xlim(_ax.get_xlim())
    # _ax2.set_xticks(channels)
    # _ax2.set_xticklabels([str(ch) for ch in neu_channels], rotation=90, fontsize=10)
    # _ax2.set_xlabel('NEU TPC Channel', fontsize=14)
    plt.tight_layout()
    plt.show()
    return (noise_rms,)


@app.cell
def _(colors, np, plt, rms, run_number):
    _fig, _ax = plt.subplots(figsize=(8, 5))

    # convert to numpy array if dict
    if isinstance(rms, dict):
        _rms_values = np.array([rms[ch] for ch in sorted(rms.keys())])
    else:
        _rms_values = np.array(rms)

    # remove NaNs
    _rms_values = _rms_values[np.isfinite(_rms_values)]

    _ax.hist(_rms_values, bins=30, color=colors[9], alpha=0.7)
    _ax.axvline(np.mean(_rms_values), color='red', ls='-', lw=1.5, label=f'Mean: {np.mean(_rms_values):.3f}')
    _ax.set_xlabel('Noise Std (ADC Counts)')
    _ax.set_ylabel('Number of Channels')
    _ax.set_title(f'Noise Std Distribution for Run {run_number}')
    _ax.legend()
    _ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Mean RMS: {np.mean(_rms_values):.4f}")
    print(f"Median RMS: {np.median(_rms_values):.4f}")
    print(f"Std of RMS: {np.std(_rms_values):.4f}")
    print(f"Min RMS: {np.min(_rms_values):.4f} (Channel {list(rms.keys())[np.argmin(_rms_values)]})")
    print(f"Max RMS: {np.max(_rms_values):.4f} (Channel {list(rms.keys())[np.argmax(_rms_values)]})")
    return


@app.cell
def _(get_raw_data, np):
    # overlay noise to compare
    # /NAS/ColumbiaIntegration/
    num_files1 = 1
    run_number1 = '673'

    files1 = []
    for _i in np.arange(num_files1):
        # files.append(f"/NAS/ColumbiaIntegration/readout_data/pGRAMS_bin_{run_number1}_{i}.dat")
        files1.append(f"/home/pgrams/data/sabertooth2_data/data/readout_data/pGRAMS_bin_{run_number1}_{_i}.dat")

    readout_df1 = get_raw_data.get_event_data(files=files1, light_slot=16, use_charge_roi=False, channel_threshold=[2055]*192)
    return readout_df1, run_number1


@app.cell
def _(channel_range, hf, np, readout_df1, select_hit_events):
    counts1 = {_ch: [] for _ch in range(channel_range[0], channel_range[1])}
    for _evt in range(0, len(readout_df1) - 2):
        if readout_df1['charge_adc_words'][_evt].ndim < 2:
            continue

        _num_channels = readout_df1['charge_adc_words'][_evt].shape[0]
        if _num_channels < channel_range[1]: 
            print(f'event {_evt} has only {_num_channels} channels...') 
            continue

        if hf.are_hits(readout_df=readout_df1, event=_evt) and select_hit_events:
            continue

        for _ch in np.arange(channel_range[0], channel_range[1]):
            counts1[_ch].append(readout_df1['charge_adc_words'][_evt][_ch, 0:125])
    return (counts1,)


@app.cell
def _(counts1, noise_hist, np, plt):
    baselines1, rms1= noise_hist(counts1, channels=np.arange(0,192), baseline=470, range=40, num_bins=40, subplot=6)
    plt.show()
    return baselines1, rms1


@app.cell
def _(baseline, baselines1, channels, colors, np, plt, rms1):
    baseline1 = [baselines1[ch] for ch in channels]
    noise_rms1 = [rms1[ch] for ch in channels]


    bar_width = 0.4
    x = np.arange(len(channels))

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.grid(which='both', visible='True', lw=0.2, ls='--', alpha=0.5)

    # Plot the first bar set, shifted left
    bars1 = _ax.bar(x + bar_width/2, baseline, width=bar_width, color=colors[9], alpha=0.4, label='Full Readout Chain',)
                                # yerr=noise_rms, capsize=2, error_kw=dict(ecolor='royalblue', alpha=0.7,  elinewidth=1, capthick=1))
    # Plot the second bar set, shifted right
    bars2 = _ax.bar(x - bar_width/2, baseline1, width=bar_width, color=colors[9], alpha=0.7, label='ADC Only',)
                                # yerr=noise_rms1, capsize=2, error_kw=dict(ecolor='royalblue', alpha=0.7, elinewidth=1, capthick=1))

    _ax.set_xlabel('CU ADC Channel', fontsize=8)
    _ax.set_ylabel('Baseline (ADC Counts)', fontsize=14)
    # _ax.set_xticks(x)
    # _ax.set_xticklabels([str(ch) for ch in channels], rotation=90, fontsize=10)

    # Add legend
    _ax.legend(fontsize=12)


    # # Add secondary x-axis
    # _ax2 = _ax.twiny()
    # _ax2.set_xlim(_ax.get_xlim())
    # _ax2.set_xticks(x)
    # _ax2.set_xticklabels([str(ch) for ch in neu_channels], rotation=90, fontsize=10)
    # _ax2.set_xlabel('NEU TPC Channel', fontsize=8)
    # plt.tight_layout()

    # plt.title('Baselines+RMS -- TPC HV on')

    plt.ylim(440, 500)
    plt.show()
    return bar_width, noise_rms1, x


@app.cell
def _(
    bar_width,
    colors,
    noise_rms,
    noise_rms1,
    plt,
    run_number,
    run_number1,
    x,
):
    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.grid(which='both', visible='True', lw=0.2, ls='--', alpha=0.5)

    # Plot the first bar set, shifted left
    _bars1 = _ax.bar(x + bar_width/2, noise_rms, width=bar_width, color=colors[2], alpha=0.4, label=f'Run {run_number}')
    # Plot the second bar set, shifted right
    _bars2 = _ax.bar(x - bar_width/2, noise_rms1, width=bar_width, color=colors[2], alpha=0.7, label=f'Run {run_number1}')

    _ax.set_xlabel('CU ADC Channel', fontsize=14)
    _ax.set_ylabel('Noise Standard Deviation (ADC Counts)', fontsize=14)
    # _ax.set_xticks(x)
    # _ax.set_xticklabels([str(ch) for ch in channels], rotation=90, fontsize=10)

    # Add legend
    _ax.legend(fontsize=12)

    # Add secondary x-axis
    # _ax2 = _ax.twiny()
    # _ax2.set_xlim(_ax.get_xlim())
    # _ax2.set_xticks(x)
    # _ax2.set_xticklabels([str(ch) for ch in neu_channels], rotation=90, fontsize=10)
    # _ax2.set_xlabel('NEU TPC Channel', fontsize=14)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(colors, np, plt, rms, rms1, run_number1):
    _fig, _ax = plt.subplots(figsize=(8, 5))

    # Convert to numpy array if it's a dictionary
    if isinstance(rms, dict):
        _rms_values = np.array([rms1[ch] for ch in sorted(rms.keys())])
    else:
        _rms_values = np.array(rms1)

    # Remove NaNs if any
    _rms_values = _rms_values[np.isfinite(_rms_values)]

    _ax.hist(_rms_values, bins=30, color=colors[9], alpha=0.7)
    _ax.axvline(np.mean(_rms_values), color='red', ls='-', lw=1.5, label=f'Mean: {np.mean(_rms_values):.3f}')
    _ax.set_xlabel('Noise Std (ADC Counts)')
    _ax.set_ylabel('Number of Channels')
    _ax.set_title(f'Noise Std Distribution for Run {run_number1}')
    _ax.legend()
    _ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Mean RMS: {np.mean(_rms_values):.4f}")
    print(f"Median RMS: {np.median(_rms_values):.4f}")
    print(f"Std of RMS: {np.std(_rms_values):.4f}")
    print(f"Min RMS: {np.min(_rms_values):.4f} (Channel {list(rms.keys())[np.argmin(_rms_values)]})")
    print(f"Max RMS: {np.max(_rms_values):.4f} (Channel {list(rms.keys())[np.argmax(_rms_values)]})")
    return


@app.cell
def _(colors, noise_rms, noise_rms1, np, plt, run_number, run_number1):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # convert to numpy arrays
    _rms1 = np.array(noise_rms)
    _rms2 = np.array(noise_rms1)
    _channels = np.arange(len(_rms1))

    _residuals = _rms1 - _rms2

    # residuals histogram
    _ax1 = _axes[0]
    _ax1.hist(_residuals, bins=30, color=colors[2], alpha=0.7)
    _ax1.axvline(0, color='red', ls='--', lw=1.5, label='$\Delta$ = 0')
    _ax1.axvline(np.mean(_residuals), color='#ee8700', ls='-', lw=1.5, label=f'Mean = {np.mean(_residuals):.4f}')
    _ax1.axvline(np.median(_residuals), color='#00583a', ls='-', lw=1.5, label=f'Median = {np.median(_residuals):.4f}')
    _ax1.set_xlabel(f'$\Delta$ STD (Run {run_number} - Run {run_number1})')
    _ax1.set_ylabel('Count')
    _ax1.set_title('Residuals Distribution')
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)

    # residuals vs channel
    _ax2 = _axes[1]
    _ax2.bar(_channels, _residuals, color=colors[0], alpha=0.7)
    _ax2.axhline(0, color='red', ls='--', lw=1)
    _ax2.axhline(np.mean(_residuals), color='#ee8700', ls='-', lw=1, label=f'Mean: {np.mean(_residuals):.4f}')
    _ax2.set_xlabel('Channel')
    _ax2.set_ylabel(f'$\Delta$ STD (Run {run_number} - Run {run_number1})')
    _ax2.set_title('Residuals vs Channel')
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # summary stats
    print("=== Summary Statistics ===")
    print(f"Mean residual: {np.mean(_residuals):.4f}")
    print(f"Std of residuals: {np.std(_residuals):.4f}")
    print(f"Max positive diff: Channel {_channels[np.argmax(_residuals)]} ({np.max(_residuals):.4f})")
    print(f"Max negative diff: Channel {_channels[np.argmin(_residuals)]} ({np.min(_residuals):.4f})")
    return


@app.cell
def _(colors, noise_rms, noise_rms1, np, plt, run_number, run_number1):
    _fig, _ax = plt.subplots(figsize=(12, 5))

    # Convert to numpy arrays if they're dictionaries
    _rms1 = np.array(noise_rms)
    _rms2 = np.array(noise_rms1)
    _channels = np.arange(len(_rms1))

    _pct_diff = 100 * (_rms1 - _rms2) / _rms2

    _ax.bar(_channels, _pct_diff, color=colors[1], alpha=0.7)
    _ax.axhline(0, color='red', ls='--', lw=1)
    _ax.axhline(np.mean(_pct_diff), color='orange', ls='-', lw=1, label=f'Mean: {np.mean(_pct_diff):.2f}\\%')
    _ax.set_xlabel('Channel')
    _ax.set_ylabel(r'Percent Difference (\%)')
    _ax.set_title(f'Percent Difference: (Run {run_number} - Run {run_number1}) / Run{run_number1} x 100')
    _ax.legend()
    _ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- Top 10 most different channels ---
    _sorted_idx = np.argsort(np.abs(_pct_diff))[::-1]
    print("=== Top 10 Most Different Channels ===")
    print(f"{'Channel':<10} {'Run 774':<12} {'Run 673':<12} {'Diff':<12} {'Pct Diff':<10}")
    for _idx in _sorted_idx[:10]:
        print(f"{_channels[_idx]:<10} {_rms1[_idx]:<12.4f} {_rms2[_idx]:<12.4f} {(_rms1[_idx]-_rms2[_idx]):<12.4f} {_pct_diff[_idx]:<10.2f}%")
    return


if __name__ == "__main__":
    app.run()
