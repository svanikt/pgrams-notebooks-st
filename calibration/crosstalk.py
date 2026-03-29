import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    ## Charge Channel Crosstalk
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
    import pandas as pd
    from scipy.optimize import curve_fit

    colors = sns.color_palette('colorblind')
    # sns.set_theme()

    plt.style.use('default')
    plt.style.use('/home/pgrams/latex-cm.mplstyle')

    abs_repo_path = os.path.abspath('/home/pgrams/daq_analysis/PGramsRawData')
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
    import raw_data_ana.hit_finding as hf

    return get_raw_data, hf


@app.cell
def _(mo):
    mo.md(r"""
    ## Configuration
    Set the injection channel, neighboring channels to inspect, and the two run numbers to compare.
    """)
    return


@app.cell
def _():
    PULSE_CHANNEL  = 28              # channel receiving the injected pulse
    NEIGHBORS      = list(range(0, 30))  # channels to inspect (include PULSE_CHANNEL)
    BASELINE_REGION = (0, 20)       # sample indices [start, end) used to measure per-channel baseline
    TIMESIZE       = 255             # readout window size in 2 MHz ticks

    run_number_A  = '766'     # long L-com  white cable
    run_number_B  = '765'     # short cable leader cables
    # run_number_B  = '764'     # short L-com cables
    label_A       = f'Run {run_number_A}: Original L-Com Cable'
    label_B       = f'Run {run_number_B}: Short L-Com Cable'
    return (
        BASELINE_REGION,
        NEIGHBORS,
        PULSE_CHANNEL,
        TIMESIZE,
        label_A,
        label_B,
        run_number_A,
        run_number_B,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Load Raw Binary Data
    """)
    return


@app.cell
def _(get_raw_data, np, run_number_A):
    files_A = [f"/home/pgrams/data/sabertooth2_data/data/readout_data/pGRAMS_bin_{run_number_A}_{i}.dat"
               for i in np.arange(1)]
    readout_df_A = get_raw_data.get_event_data(files=files_A, light_slot=16, use_charge_roi=False, channel_threshold=[2055]*192)
    return (readout_df_A,)


@app.cell
def _(get_raw_data, np, run_number_B):
    files_B = [f"/home/pgrams/data/sabertooth2_data/data/readout_data/pGRAMS_bin_{run_number_B}_{i}.dat"
               for i in np.arange(1)]
    readout_df_B = get_raw_data.get_event_data(files=files_B, light_slot=16, use_charge_roi=False, channel_threshold=[2055]*192)
    return (readout_df_B,)


@app.cell
def _(NEIGHBORS, TIMESIZE, hf, np, plt, readout_df_A):
    # Show first event that hf.are_hits() flags — so you can see what's being excluded
    _bad_evt = None
    for _evt in range(len(readout_df_A)):
        if readout_df_A['charge_adc_words'][_evt].ndim < 2:
            continue
        if hf.are_hits(readout_df=readout_df_A, event=_evt, jump=8):
            _bad_evt = _evt
            break

    if _bad_evt is not None:
        _wf = readout_df_A['charge_adc_words'][_bad_evt]
        _xaxis = np.linspace(-TIMESIZE / 2, TIMESIZE, _wf.shape[1])
        _, _ax = plt.subplots(figsize=(12, 4))
        for _ch in NEIGHBORS:
            if _ch < _wf.shape[0] and  _ch==0:
                _ax.plot(_xaxis, _wf[_ch, :].astype(float), lw=0.8, label=f'Ch {_ch}')
        _ax.set_xlabel(r'[$\mu$s]')
        _ax.set_ylabel('ADC Counts (raw)')
        _ax.set_title(f'Example flagged event (evt {_bad_evt}) — excluded from baseline')
        _ax.legend(fontsize=7, ncol=5)
        plt.tight_layout()
        plt.show()
    else:
        print("No flagged events found in run A.")
    return


@app.cell
def _(hf, np):
    def compute_baselines(readout_df, channels, sample_region):
        """
        Measure per-channel baseline as mean ± std over waveform samples
        [sample_region[0] : sample_region[1]], pooled across all events.

        Events flagged by hf.are_hits() (baseline dropouts, large swings) are
        excluded before accumulating samples, matching the chargebaselines.py approach.

        Returns
        -------
        means : dict {ch: float}  — baseline (ADC counts) for each channel
        stds  : dict {ch: float}  — noise std (ADC counts) for each channel
        """
        all_samples = {ch: [] for ch in channels}

        for evt in range(len(readout_df)):
            wf = readout_df['charge_adc_words'][evt]
            if wf.ndim < 2 or wf.shape[0] <= max(channels):
                continue
            if hf.are_hits(readout_df=readout_df, event=evt):
                continue
            for ch in channels:
                all_samples[ch].append(
                    wf[ch, sample_region[0]:sample_region[1]].astype(float)
                )

        means = {}
        stds  = {}
        for ch in channels:
            if all_samples[ch]:
                data = np.concatenate(all_samples[ch])
                means[ch] = np.mean(data)
                stds[ch]  = np.std(data)
            else:
                means[ch] = np.nan
                stds[ch]  = np.nan

        return means, stds

    return (compute_baselines,)


@app.cell
def _(BASELINE_REGION, NEIGHBORS, compute_baselines, label_A, readout_df_A):
    _channels = list(set(NEIGHBORS))
    baselines_A, baselines_std_A = compute_baselines(readout_df_A, _channels, BASELINE_REGION)
    print(f"{label_A} — baselines measured over samples {BASELINE_REGION[0]}-{BASELINE_REGION[1]}:")
    for _ch in sorted(_channels):
        print(f"  Ch {_ch}: {baselines_A[_ch]:.2f} ± {baselines_std_A[_ch]:.2f} ADC")
    return (baselines_A,)


@app.cell
def _(BASELINE_REGION, NEIGHBORS, compute_baselines, label_B, readout_df_B):
    _channels = list(set(NEIGHBORS))
    baselines_B, baselines_std_B = compute_baselines(readout_df_B, _channels, BASELINE_REGION)
    print(f"{label_B} — baselines measured over samples {BASELINE_REGION[0]}–{BASELINE_REGION[1]}:")
    for _ch in sorted(_channels):
        print(f"  Ch {_ch}: {baselines_B[_ch]:.2f} ± {baselines_std_B[_ch]:.2f} ADC")
    return (baselines_B,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preview Events — Run A
    Overlay the injection channel and its neighbors to confirm the pulse shape and spot any induced signals.
    """)
    return


@app.cell
def _(
    NEIGHBORS,
    PULSE_CHANNEL,
    TIMESIZE,
    baselines_A,
    colors,
    label_A,
    np,
    plt,
    readout_df_A,
):
    from matplotlib.ticker import AutoMinorLocator

    for _evt in range(0, 5):
        _wf = readout_df_A['charge_adc_words'][_evt]
        if _wf.ndim < 2:
            continue

        _fig, _ax = plt.subplots(figsize=(12, 4))
        _xaxis = np.linspace(-TIMESIZE / 2, TIMESIZE, _wf.shape[1])

        for _i, _ch in enumerate(NEIGHBORS):
            _lw = 2.0 if _ch == PULSE_CHANNEL else 1.0
            _ls = '-'  if _ch == PULSE_CHANNEL else '--'
            _label = f'Ch {_ch} (injection)' if _ch == PULSE_CHANNEL else f'Ch {_ch}'
            _ax.plot(_xaxis, _wf[_ch, :].astype(float) - baselines_A[_ch],
                     color=colors[_i % len(colors)], lw=_lw, ls=_ls, label=_label)

        _ax.set_xlim(-50, 50)
        _ax.set_ylim(0,20)
        _ax.set_xlabel(r'[$\mu$s]')
        _ax.set_ylabel('Baseline-Subtracted ADC Counts')
        _ax.set_title(f'{label_A}, Event {_evt}')
        _ax.legend(loc='upper right', fontsize=9, ncol=3)
        _ax.xaxis.set_minor_locator(AutoMinorLocator())
        _ax.yaxis.set_minor_locator(AutoMinorLocator())
        _ax.grid(True, which='major', linestyle='-', linewidth=0.8)
        _ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()
    return (AutoMinorLocator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preview Events — Run B
    """)
    return


@app.cell
def _(
    AutoMinorLocator,
    NEIGHBORS,
    PULSE_CHANNEL,
    TIMESIZE,
    baselines_B,
    colors,
    label_B,
    np,
    plt,
    readout_df_B,
):
    for _evt in range(0, 5):
        _wf = readout_df_B['charge_adc_words'][_evt]
        if _wf.ndim < 2:
            continue

        _fig, _ax = plt.subplots(figsize=(12, 4))
        _xaxis = np.linspace(-TIMESIZE / 2, TIMESIZE, _wf.shape[1])

        for _i, _ch in enumerate(NEIGHBORS):
            _lw = 2.0 if _ch == PULSE_CHANNEL else 1.0
            _ls = '-'  if _ch == PULSE_CHANNEL else '--'
            _label = f'Ch {_ch} (injection)' if _ch == PULSE_CHANNEL else f'Ch {_ch}'
            _ax.plot(_xaxis, _wf[_ch, :].astype(float) - baselines_B[_ch],
                     color=colors[_i % len(colors)], lw=_lw, ls=_ls, label=_label)

        # _ax.set_xlim(-128, 156)
        _ax.set_xlim(-50, 50)
        _ax.set_ylim(0,20)
        _ax.set_xlabel(r'[$\mu$s]')
        _ax.set_ylabel('Baseline-Subtracted ADC Counts')
        _ax.set_title(f'{label_B}, Event {_evt}')
        _ax.legend(loc='upper right', fontsize=9, ncol=3)
        _ax.xaxis.set_minor_locator(AutoMinorLocator())
        _ax.yaxis.set_minor_locator(AutoMinorLocator())
        _ax.grid(True, which='major', linestyle='-', linewidth=0.8)
        _ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Measure Crosstalk

    For each event, the peak sample in the injection channel is located.
    The baseline-subtracted amplitude at that same sample in every neighboring channel
    gives the induced signal. The crosstalk fraction for channel $n$ is:

    $$\text{XT}_n = \frac{A_n}{A_\text{inj}}$$

    Per-channel baselines are measured from the pre-signal region defined by `BASELINE_REGION`.
    """)
    return


@app.cell
def _(NEIGHBORS, PULSE_CHANNEL, np):
    from scipy.signal import find_peaks

    HIT_HEIGHT = 5   # min peak height above baseline in injection channel
    HIT_PROM   = 2   # min prominence

    def measure_crosstalk(readout_df, baselines, sample_range=None):
        """
        Returns (primary_amps, neighbor_amps) where:
          primary_amps  — 1D array of peak amplitudes in PULSE_CHANNEL, one per qualifying event
          neighbor_amps — dict {ch: 1D array} of co-timed amplitudes in each neighbor

        :param readout_df: pandas DataFrame with charge_adc_words
        :param baselines: dict {ch: float} of per-channel measured baselines
        :param sample_range: tuple (start, end) sample indices to search for peaks.
                             None means search the full waveform.
        """
        primary_amps = []
        neighbor_amps = {ch: [] for ch in NEIGHBORS if ch != PULSE_CHANNEL}

        for evt in range(len(readout_df)):
            wf = readout_df['charge_adc_words'][evt]
            if wf.ndim < 2 or wf.shape[0] <= max(NEIGHBORS):
                continue

            signal = wf[PULSE_CHANNEL, :].astype(float) - baselines[PULSE_CHANNEL]

            # Find peaks in the specified sample range
            if sample_range is not None:
                n_samples = wf.shape[1]
                start = sample_range[0] % n_samples
                end = sample_range[1] % n_samples if sample_range[1] != 0 else n_samples
                signal_window = signal[start:end]
                peaks, _ = find_peaks(signal_window, height=HIT_HEIGHT, prominence=HIT_PROM)
                if len(peaks) == 0:
                    continue
                peak_idx = peaks[np.argmax(signal_window[peaks])] + start
            else:
                peaks, _ = find_peaks(signal, height=HIT_HEIGHT, prominence=HIT_PROM)
                if len(peaks) == 0:
                    continue
                peak_idx = peaks[np.argmax(signal[peaks])]

            primary_amps.append(signal[peak_idx])

            for ch in NEIGHBORS:
                if ch == PULSE_CHANNEL:
                    continue
                neighbor_amps[ch].append(wf[ch, peak_idx].astype(float) - baselines[ch])

        return np.array(primary_amps), {ch: np.array(v) for ch, v in neighbor_amps.items()}


    return (measure_crosstalk,)


@app.cell
def _(
    baselines_A,
    baselines_B,
    label_A,
    label_B,
    measure_crosstalk,
    np,
    readout_df_A,
    readout_df_B,
):
    primary_A, neighbor_A = measure_crosstalk(readout_df_A, baselines_A, sample_range=[175, 225])
    primary_B, neighbor_B = measure_crosstalk(readout_df_B, baselines_B, sample_range=[175, 225])
    print(f"{label_A}: {len(primary_A)} events  |  primary amp mean={np.mean(primary_A):.1f}  std={np.std(primary_A):.1f} ADC")
    print(f"{label_B}: {len(primary_B)} events  |  primary amp mean={np.mean(primary_B):.1f}  std={np.std(primary_B):.1f} ADC")
    return neighbor_A, neighbor_B, primary_A, primary_B


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Crosstalk Fraction Per Neighbor — Run Comparison
    Mean crosstalk fraction (%) for each neighboring channel, with runs A and B side by side.
    """)
    return


@app.cell
def _(
    NEIGHBORS,
    PULSE_CHANNEL,
    colors,
    label_A,
    label_B,
    neighbor_A,
    neighbor_B,
    np,
    plt,
    primary_A,
    primary_B,
):
    _neighbor_chs = [ch for ch in NEIGHBORS if ch != PULSE_CHANNEL]

    _mean_A, _err_A = [], []
    _mean_B, _err_B = [], []

    for _ch in _neighbor_chs:
        _xt_A = neighbor_A[_ch] / primary_A * 100
        _xt_B = neighbor_B[_ch] / primary_B * 100
        _mean_A.append(np.mean(_xt_A));  _err_A.append(np.std(_xt_A) / np.sqrt(len(_xt_A)))
        _mean_B.append(np.mean(_xt_B));  _err_B.append(np.std(_xt_B) / np.sqrt(len(_xt_B)))

    _mean_A = np.array(_mean_A)
    _mean_B = np.array(_mean_B)
    _err_A  = np.array(_err_A)
    _err_B  = np.array(_err_B)

    _x = np.arange(len(_neighbor_chs))
    _w = 0.35

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.bar(_x - _w/2, _mean_A, _w, color=colors[0], alpha=0.7, label=label_A)
            # yerr=_err_A, capsize=4, error_kw=dict(elinewidth=1, capthick=1))
    _ax.bar(_x + _w/2, _mean_B, _w, color=colors[1], alpha=0.7, label=label_B)
            # yerr=_err_B, capsize=4, error_kw=dict(elinewidth=1, capthick=1))
    _ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    _ax.set_xticks(_x)
    _ax.set_xticklabels([f'Ch {ch}' for ch in _neighbor_chs], rotation=45)
    _ax.set_xlabel('Neighbor Channel')
    _ax.set_ylabel(r'Crosstalk Fraction (\%)')
    _ax.set_title(f'Crosstalk Fraction (injection on Ch {PULSE_CHANNEL})')
    _ax.legend(fontsize=12)
    _ax.grid(which='major', linestyle='--', linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    plt.show()

    # tabulate
    print(f"{'Channel':<10} {label_A+' XT (%)':>16} {'±':>3} {'err':>6}    {label_B+' XT (%)':>16} {'±':>3} {'err':>6}")
    for _ch, _mA, _eA, _mB, _eB in zip(_neighbor_chs, _mean_A, _err_A, _mean_B, _err_B):
        print(f"  Ch {_ch:<6} {_mA:>16.3f} {'±':>3} {_eA:<6.3f}    {_mB:>16.3f} {'±':>3} {_eB:<6.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Event-by-Event Crosstalk Distributions

    All subplots share the same x-axis, so noise-floor channels (zero-centred Gaussian)
    are immediately distinguishable from channels with real crosstalk (distribution
    shifted away from zero).  The dashed grey line marks zero; solid coloured lines
    mark each run's mean.
    """)
    return


@app.cell
def _(
    NEIGHBORS,
    PULSE_CHANNEL,
    colors,
    label_A,
    label_B,
    neighbor_A,
    neighbor_B,
    np,
    plt,
    primary_A,
    primary_B,
):
    _neighbor_chs = [ch for ch in NEIGHBORS if ch != PULSE_CHANNEL]
    _ncols = 5
    _nrows = len(_neighbor_chs) // _ncols + (len(_neighbor_chs) % _ncols > 0)

    # Build all distributions up front so we can derive a single shared x range
    _xt = {_ch: (neighbor_A[_ch] / primary_A * 100,
                 neighbor_B[_ch] / primary_B * 100)
           for _ch in _neighbor_chs}

    _all_vals = np.concatenate([v for pair in _xt.values() for v in pair])
    _xlo  = np.percentile(_all_vals, 0.5)
    _xhi  = np.percentile(_all_vals, 99.5)
    _bins = np.linspace(_xlo, _xhi, 60)

    _fig, _axs = plt.subplots(_nrows, _ncols, figsize=(4 * _ncols, 3 * _nrows),
                               sharex=True, sharey=False)
    _axs = _axs.flatten()

    for _i, _ch in enumerate(_neighbor_chs):
        _ax = _axs[_i]
        _xt_A, _xt_B = _xt[_ch]

        _ax.hist(_xt_A, bins=_bins, histtype='stepfilled', color=colors[0], alpha=0.4)
        _ax.hist(_xt_B, bins=_bins, histtype='stepfilled', color=colors[1], alpha=0.4)
        _ax.hist(_xt_A, bins=_bins, histtype='step', color=colors[0], lw=1.2, label=label_A)
        _ax.hist(_xt_B, bins=_bins, histtype='step', color=colors[1], lw=1.2, label=label_B)

        _ax.axvline(0, color='gray', ls='--', lw=0.8)
        _ax.axvline(np.mean(_xt_A), color=colors[0], ls='-', lw=1.4)
        _ax.axvline(np.mean(_xt_B), color=colors[1], ls='-', lw=1.4)

        _ax.set_title(f'Ch {_ch}', fontsize=9)
        _ax.set_xlabel(r'XT (\%)', fontsize=8)
        if _i % _ncols == 0:
            _ax.set_ylabel('Events', fontsize=8)
        _ax.tick_params(labelsize=7)

    _axs[0].legend(fontsize=6, loc='upper right')

    for _j in range(len(_neighbor_chs), len(_axs)):
        _axs[_j].set_visible(False)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
