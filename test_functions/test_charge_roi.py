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
    import pandas as pd
    import matplotlib.pyplot as plt

    # The root directory of the code is up 2 directories
    abs_repo_path = os.path.abspath('/home/pgrams/tpc_data/software/PGramsRawData')
    # Insert the path to the front of sys.path if it is not already there
    if not abs_repo_path in sys.path:
        sys.path.insert(0, abs_repo_path)
    return mo, pd, plt


@app.cell
def _():
    import decoder_bindings_roi
    import raw_data_ana.data_checks as dc
    return dc, decoder_bindings_roi


@app.cell
def _():
    channel_threshold = [2054] * 64
    return (channel_threshold,)


@app.cell
def _(channel_threshold, decoder_bindings_roi):
    ge = decoder_bindings_roi.ProcessEvents(light_slot=16, use_charge_roi=True,channel_threshold=channel_threshold)
    return (ge,)


@app.cell
def _(ge):
    files = "/home/pgrams/data/readout_data/pGRAMS_bin_156_0.dat"
    ge.open_file(files)
    return


@app.cell
def _(ge, pd):
    # for i in range(30):
    l = []
    while ge.get_event():
        tmp = ge.get_event_dict()
        l.append(tmp)
    df = pd.DataFrame(l)
    return (df,)


@app.cell
def _(dc, df):
    dc.check_fems(readout_df=df)
    dc.word_count_check(readout_df=df)
    return


@app.cell
def _(df):
    # df['charge_adc_words'][49]
    # df['charge_idx'][49]
    df[49:51]
    # print(df['charge_channel'][49])
    # df['charge_adc_words'][49].shape
    return


@app.cell
def _(df, plt):
    evt = 49

    for evt in range(1000):
        if len(df['charge_channel'][evt]) > 0: plt.figure(figsize=(12,5))
        for i, (ch, idx, roi) in enumerate(zip(df['charge_channel'][evt], df['charge_adc_idx'][evt], df['charge_adc_words'][evt])):
            plt.plot(idx, roi, marker='.', label=str(ch))
        if len(df['charge_channel'][evt]) > 0:
            plt.title("Event: " + str(evt))
            plt.legend()
            plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
