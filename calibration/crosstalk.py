import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    ## Charge Channel Crosstalk
    A pulse is injected onto a single known channel. We measure whether neighboring channels
    pick up the pulse in a measurable way, and compare the result across two runs.
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

    return (get_raw_data,)


@app.cell
def _(np):
    def gaussian(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Configuration
    Set the injection channel, neighboring channels to inspect, and the two run numbers to compare.
    """)
    return


@app.cell
def _():
    PULSE_CHANNEL = 29        # channel receiving the injected pulse
    NEIGHBORS     = list(range(25, 34))  # channels to inspect (include PULSE_CHANNEL)
    BASELINE      = 470       # nominal ADC baseline — update to match your runs
    TIMESIZE      = 255       # readout window size in 2 MHz ticks
    HIT_HEIGHT    = 15        # min peak height above baseline in injection channel
    HIT_PROM      = 10        # min prominence

    run_number_A  = '766'     # e.g. shielded cable
    run_number_B  = '765'     # e.g. ribbon cable (or whatever configuration B is)
    label_A       = f'Run {run_number_A}'
    label_B       = f'Run {run_number_B}'
    return (
        BASELINE,
        HIT_HEIGHT,
        HIT_PROM,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preview Events — Run A
    Overlay the injection channel and its neighbors to confirm the pulse shape and spot any induced signals.
    """)
    return


@app.cell
def _(
    BASELINE,
    NEIGHBORS,
    PULSE_CHANNEL,
    TIMESIZE,
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

        _fig, _ax = plt.subplots(figsize=(16, 4))
        _xaxis = np.linspace(-TIMESIZE / 2, TIMESIZE, _wf.shape[1])

        for _i, _ch in enumerate(NEIGHBORS):
            _lw = 2.0 if _ch == PULSE_CHANNEL else 1.0
            _ls = '-'  if _ch == PULSE_CHANNEL else '--'
            _label = f'Ch {_ch} (injection)' if _ch == PULSE_CHANNEL else f'Ch {_ch}'
            _ax.plot(_xaxis, _wf[_ch, :].astype(float) - BASELINE,
                     color=colors[_i % len(colors)], lw=_lw, ls=_ls, label=_label)

        _ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        _ax.axvline(-128, color='red', linestyle='--', linewidth=0.8)
        _ax.axvline(0,    color='red', linestyle='--', linewidth=0.8)
        _ax.set_xlim(-128, 156)
        _ax.set_xlabel(r'[$\mu$s]')
        _ax.set_ylabel('Charge [ADC $-$ baseline]')
        _ax.set_title(f'{label_A} — Event {_evt}')
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
    BASELINE,
    NEIGHBORS,
    PULSE_CHANNEL,
    TIMESIZE,
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

        _fig, _ax = plt.subplots(figsize=(16, 4))
        _xaxis = np.linspace(-TIMESIZE / 2, TIMESIZE, _wf.shape[1])

        for _i, _ch in enumerate(NEIGHBORS):
            _lw = 2.0 if _ch == PULSE_CHANNEL else 1.0
            _ls = '-'  if _ch == PULSE_CHANNEL else '--'
            _label = f'Ch {_ch} (injection)' if _ch == PULSE_CHANNEL else f'Ch {_ch}'
            _ax.plot(_xaxis, _wf[_ch, :].astype(float) - BASELINE,
                     color=colors[_i % len(colors)], lw=_lw, ls=_ls, label=_label)

        _ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        _ax.axvline(-128, color='red', linestyle='--', linewidth=0.8)
        _ax.axvline(0,    color='red', linestyle='--', linewidth=0.8)
        _ax.set_xlim(-128, 156)
        _ax.set_xlabel(r'[$\mu$s]')
        _ax.set_ylabel('Charge [ADC $-$ baseline]')
        _ax.set_title(f'{label_B} — Event {_evt}')
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
    """)
    return


@app.cell
def _(BASELINE, HIT_HEIGHT, HIT_PROM, NEIGHBORS, PULSE_CHANNEL, np):
    from scipy.signal import find_peaks as _fp

    def measure_crosstalk(readout_df):
        """
        Returns (primary_amps, neighbor_amps) where:
          primary_amps  — 1D array of peak amplitudes in PULSE_CHANNEL, one per qualifying event
          neighbor_amps — dict {ch: 1D array} of co-timed amplitudes in each neighbor
        """
        _primary_amps = []
        _neighbor_amps = {ch: [] for ch in NEIGHBORS if ch != PULSE_CHANNEL}

        for _evt in range(len(readout_df)):
            _wf = readout_df['charge_adc_words'][_evt]
            if _wf.ndim < 2 or _wf.shape[0] <= max(NEIGHBORS):
                continue

            _signal = _wf[PULSE_CHANNEL, :].astype(float) - BASELINE
            _peaks, _ = _fp(_signal, height=HIT_HEIGHT, prominence=HIT_PROM)
            if len(_peaks) == 0:
                continue

            _peak_idx = _peaks[np.argmax(_signal[_peaks])]
            _primary_amps.append(_signal[_peak_idx])

            for _ch in NEIGHBORS:
                if _ch == PULSE_CHANNEL:
                    continue
                _neighbor_amps[_ch].append(_wf[_ch, _peak_idx].astype(float) - BASELINE)

        return np.array(_primary_amps), {ch: np.array(v) for ch, v in _neighbor_amps.items()}

    return (measure_crosstalk,)


@app.cell
def _(label_A, label_B, measure_crosstalk, np, readout_df_A, readout_df_B):
    primary_A, neighbor_A = measure_crosstalk(readout_df_A)
    primary_B, neighbor_B = measure_crosstalk(readout_df_B)

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

    _mean_A = np.array(_mean_A) * 1   # already in %
    _mean_B = np.array(_mean_B) * 1
    _err_A  = np.array(_err_A)
    _err_B  = np.array(_err_B)

    _x = np.arange(len(_neighbor_chs))
    _w = 0.35

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.bar(_x - _w/2, _mean_A, _w, color=colors[0], alpha=0.7, label=label_A,
            yerr=_err_A, capsize=4, error_kw=dict(elinewidth=1, capthick=1))
    _ax.bar(_x + _w/2, _mean_B, _w, color=colors[1], alpha=0.7, label=label_B,
            yerr=_err_B, capsize=4, error_kw=dict(elinewidth=1, capthick=1))
    _ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    _ax.set_xticks(_x)
    _ax.set_xticklabels([f'Ch {ch}' for ch in _neighbor_chs], rotation=45)
    _ax.set_xlabel('Neighbor Channel')
    _ax.set_ylabel(r'Crosstalk Fraction (\%)')
    _ax.set_title(f'Crosstalk onto Neighboring Channels (injection on Ch {PULSE_CHANNEL})')
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
    _ncols = 4
    _nrows = len(_neighbor_chs) // _ncols + (len(_neighbor_chs) % _ncols > 0)

    _fig, _axs = plt.subplots(_nrows, _ncols, figsize=(20, 3 * _nrows))
    _axs = _axs.flatten()

    for _i, _ch in enumerate(_neighbor_chs):
        _xt_A = neighbor_A[_ch] / primary_A * 100
        _xt_B = neighbor_B[_ch] / primary_B * 100
        _all  = np.concatenate([_xt_A, _xt_B])
        _bins = np.linspace(_all.min(), _all.max(), 30)

        _axs[_i].hist(_xt_A, bins=_bins, histtype='step', color=colors[0], label=label_A)
        _axs[_i].hist(_xt_B, bins=_bins, histtype='step', color=colors[1], label=label_B)
        _axs[_i].axvline(np.mean(_xt_A), color=colors[0], ls='--', lw=1.2)
        _axs[_i].axvline(np.mean(_xt_B), color=colors[1], ls='--', lw=1.2)
        _axs[_i].set_title(f'Ch {_ch}')
        _axs[_i].set_xlabel(r'XT (\%)')
        _axs[_i].legend(fontsize=7)

    for _j in range(len(_neighbor_chs), len(_axs)):
        _axs[_j].set_visible(False)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
