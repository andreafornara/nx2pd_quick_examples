"""
Bunch-by-Bunch Luminosity Analysis

This script performs bunch-by-bunch analysis of instantaneous luminosity
for ATLAS, CMS, and LHCb experiments.

Each plot shows:
- Left: Mean luminosity evolution (averaged over colliding bunches)
- Right: Bunch-by-bunch luminosity snapshots (one point per hour, color = time)
"""

# %%
# Setup: Import packages and create spark session with large memory configuration
import nx2pd as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from nxcals.spark_session_builder import get_or_create, Flavor

# Create spark session with large memory configuration for bunch-by-bunch data
print("Creating Spark session with large memory configuration...")
spark = get_or_create(flavor=Flavor.LOCAL,
                     conf={'spark.driver.maxResultSize': '8g',
                          'spark.executor.memory':'8g',
                          'spark.driver.memory': '16g',
                          'spark.executor.instances': '20',
                          'spark.executor.cores': '2',
                          })
sk = nx.SparkIt(spark)
print("Spark session created.")

# %%
# Define the fill number and analysis parameters
fill_number = 10993

# Define time window for analysis (optional)
start_mode = 'STABLE'  # Analyze from STABLE beams
end_mode = None        # Until end of fill

# Plotting parameter: plot one snapshot every N hours
every_hours = 1

# Variables for bunch-by-bunch luminosity analysis
variables = [
    # Total luminosity
    'ATLAS:LUMI_TOT_INST',
    'CMS:LUMI_TOT_INST',
    'LHCB:LUMI_TOT_INST',

    # Bunch-by-bunch luminosity
    'ATLAS:BUNCH_LUMI_INST',
    'CMS:BUNCH_LUMI_INST',
    'LHCB:BUNCH_LUMI_INST',

    # Additional context - filling scheme
    'LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY',
    'LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY',
]

# %%
# Get the fill data
print(f"\nRetrieving bunch-by-bunch luminosity data for fill {fill_number}...")
print("This may take several minutes due to large data volume...")
df = sk.get_fill_raw_data(fill_number, variables)
print(f"Retrieved {len(df)} data points.")

# Get fill timing information
fill_info = sk.get_fill_time(fill_number)

# Determine analysis time window based on start_mode and end_mode
analysis_start = fill_info['start']
analysis_end = fill_info['end']

if start_mode is not None or end_mode is not None:
    modes = fill_info.get('modes', [])

    # Find start time
    if start_mode is not None:
        for mode_info in modes:
            if mode_info['mode'] == start_mode:
                analysis_start = mode_info['start']
                print(f"Analysis start: {start_mode} mode at {analysis_start}")
                break
        else:
            print(f"WARNING: Start mode '{start_mode}' not found, using fill start")

    # Find end time
    if end_mode is not None:
        for mode_info in modes:
            if mode_info['mode'] == end_mode:
                analysis_end = mode_info['end']
                print(f"Analysis end: {end_mode} mode at {analysis_end}")
                break
        else:
            print(f"WARNING: End mode '{end_mode}' not found, using fill end")

# Filter dataframe to analysis window
df = df[(df.index >= analysis_start) & (df.index <= analysis_end)]
print(f"Filtered to analysis window: {len(df)} data points.")

# Calculate duration and create title info
fill_start = fill_info['start']
analysis_duration_hours = (analysis_end - analysis_start).total_seconds() / 3600
fill_date_str = fill_start.strftime('%B %d, %Y at %H:%M:%S')

# Update title based on mode
if start_mode is not None:
    title_info = f"Fill {fill_number}: {start_mode} BEAMS declared on {fill_date_str}"
else:
    title_info = f"Fill {fill_number}: {fill_date_str}"

print(f"\nFill start: {fill_start}")
print(f"Analysis window start: {analysis_start}")
print(f"Analysis window end: {analysis_end}")
print(f"Analysis duration: {analysis_duration_hours:.2f} hours")

# %%
# Determine filling scheme from bunch intensity
# Data structure: Each timestamp contains a dict with 'elements' key that has the numpy array
print("\nDetermining filling scheme...")

try:
    # Get first timestamp's bunch intensity array
    b1_intensity_data = df["LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY"].dropna()
    b2_intensity_data = df["LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY"].dropna()

    # Extract the numpy array from 'elements' key
    b1_first = b1_intensity_data.iloc[0]['elements']
    b2_first = b2_intensity_data.iloc[0]['elements']

    # Find filled bunches (where intensity > 0)
    idx_b1 = np.where(b1_first > 0)[0]
    idx_b2 = np.where(b2_first > 0)[0]

    # Find colliding bunches (intersection)
    idx_b1_coll = np.intersect1d(idx_b1, idx_b2)
    idx_b2_coll = np.intersect1d(idx_b1, idx_b2)

    print(f"Beam 1: {len(idx_b1)} filled bunches")
    print(f"Beam 2: {len(idx_b2)} filled bunches")
    print(f"Colliding bunches: {len(idx_b1_coll)}")
except Exception as e:
    print(f"WARNING: Could not determine filling scheme: {e}")
    print(f"Error details: {type(e).__name__}")
    idx_b1 = np.arange(0, 100)
    idx_b2 = np.arange(0, 100)
    idx_b1_coll = np.arange(0, 100)
    idx_b2_coll = np.arange(0, 100)

# %%
# Create the combined plot (ATLAS, CMS, and LHCb)
print("\n" + "="*80)
print("Creating Bunch-by-Bunch Luminosity Plot")
print("="*80)

fig, ax = plt.subplots(figsize=(18, 22), nrows=3, ncols=2,
                       gridspec_kw={'height_ratios': [1, 1, 1], 'width_ratios': [0.5, 1]},
                       sharey=True)
plt.suptitle(title_info, fontsize=20, fontweight='bold')

# ===== ATLAS =====
# Left plot: Mean luminosity evolution
ax_left = ax[0, 0]
ax_left.set_xlabel("Time", fontsize=12)
ax_left.set_ylabel(r"$\mathcal{L}$ (10$^{35}$ m$^{-2}$ s$^{-1}$)", fontsize=12)
ax_left.set_title("ATLAS - Mean Luminosity", fontsize=13, fontweight='bold')

if 'ATLAS:BUNCH_LUMI_INST' in df.columns:
    data_atlas = df['ATLAS:BUNCH_LUMI_INST'].dropna()
    if len(data_atlas) > 0:
        # Calculate mean and std over colliding bunches (filter out zeros)
        mean_lumi = data_atlas.apply(lambda x: np.mean(x['elements'][idx_b1_coll][x['elements'][idx_b1_coll] != 0]) if isinstance(x, dict) and 'elements' in x else np.nan)
        std_lumi = data_atlas.apply(lambda x: np.std(x['elements'][idx_b1_coll][x['elements'][idx_b1_coll] != 0]) if isinstance(x, dict) and 'elements' in x else np.nan)

        # Convert from Hz/ub to 10^35 m^-2 s^-1
        mean_lumi = mean_lumi * 1e-1
        std_lumi = std_lumi * 1e-1

        ax_left.fill_between(mean_lumi.index, mean_lumi - std_lumi, mean_lumi + std_lumi,
                            color="b", alpha=0.3)
        ax_left.plot(mean_lumi, c="b", lw=2, label="ATLAS")
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(fontsize=14)

ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax_left.xaxis.set_major_locator(MaxNLocator(5))
plt.setp(ax_left.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Right plot: Bunch-by-bunch snapshots (one per hour)
ax_right = ax[0, 1]
ax_right.set_xlabel("Bunch slot (25 ns)", fontsize=12)
ax_right.set_title("ATLAS - BBB Luminosity (one snapshot/hour)", fontsize=13, fontweight='bold')
ax_right.grid(True, alpha=0.3)

if 'ATLAS:BUNCH_LUMI_INST' in df.columns:
    data_atlas = df['ATLAS:BUNCH_LUMI_INST'].dropna()
    if len(data_atlas) > 0:
        # Extract elements and convert to 10^35 m^-2 s^-1
        data_processed = data_atlas.apply(lambda x: x['elements'][idx_b1] * 1e-1 if isinstance(x, dict) and 'elements' in x else np.nan)

        total_hours = int((data_processed.index[-1] - data_processed.index[0]).total_seconds() // 3600)
        colors = cm.rainbow(np.linspace(0, 1, total_hours + 1))
        start_time = data_processed.index[0]

        # Plot one snapshot per hour
        for i in range(0, total_hours + 1, every_hours):
            time_begin = start_time + pd.Timedelta(hours=i)
            time_end = time_begin + pd.Timedelta(hours=1)
            hourly_data = data_processed[(data_processed.index >= time_begin) & (data_processed.index < time_end)]

            if len(hourly_data) > 0:
                ax_right.plot(idx_b1, hourly_data.iloc[0], '.', ms=4, c=colors[i], alpha=0.5)

# ===== CMS =====
# Left plot: Mean luminosity evolution
ax_left = ax[1, 0]
ax_left.set_xlabel("Time", fontsize=12)
ax_left.set_ylabel(r"$\mathcal{L}$ (10$^{35}$ m$^{-2}$ s$^{-1}$)", fontsize=12)
ax_left.set_title("CMS - Mean Luminosity", fontsize=13, fontweight='bold')

if 'CMS:BUNCH_LUMI_INST' in df.columns:
    data_cms = df['CMS:BUNCH_LUMI_INST'].dropna()
    if len(data_cms) > 0:
        # Calculate mean and std over colliding bunches (filter out zeros)
        mean_lumi = data_cms.apply(lambda x: np.mean(x['elements'][idx_b2_coll][x['elements'][idx_b2_coll] != 0]) if isinstance(x, dict) and 'elements' in x else np.nan)
        std_lumi = data_cms.apply(lambda x: np.std(x['elements'][idx_b2_coll][x['elements'][idx_b2_coll] != 0]) if isinstance(x, dict) and 'elements' in x else np.nan)

        # Convert from Hz/ub to 10^35 m^-2 s^-1
        mean_lumi = mean_lumi * 1e-1
        std_lumi = std_lumi * 1e-1

        ax_left.fill_between(mean_lumi.index, mean_lumi - std_lumi, mean_lumi + std_lumi,
                            color="r", alpha=0.3)
        ax_left.plot(mean_lumi, c="r", lw=2, label="CMS")
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(fontsize=14)

ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax_left.xaxis.set_major_locator(MaxNLocator(5))
plt.setp(ax_left.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Right plot: Bunch-by-bunch snapshots (one per hour)
ax_right = ax[1, 1]
ax_right.set_xlabel("Bunch slot (25 ns)", fontsize=12)
ax_right.set_title("CMS - BBB Luminosity (one snapshot/hour)", fontsize=13, fontweight='bold')
ax_right.grid(True, alpha=0.3)

if 'CMS:BUNCH_LUMI_INST' in df.columns:
    data_cms = df['CMS:BUNCH_LUMI_INST'].dropna()
    if len(data_cms) > 0:
        # Extract elements and convert to 10^35 m^-2 s^-1
        data_processed = data_cms.apply(lambda x: x['elements'][idx_b2] * 1e-1 if isinstance(x, dict) and 'elements' in x else np.nan)

        total_hours = int((data_processed.index[-1] - data_processed.index[0]).total_seconds() // 3600)
        colors = cm.rainbow(np.linspace(0, 1, total_hours + 1))
        start_time = data_processed.index[0]

        # Plot one snapshot per hour
        for i in range(0, total_hours + 1, every_hours):
            time_begin = start_time + pd.Timedelta(hours=i)
            time_end = time_begin + pd.Timedelta(hours=1)
            hourly_data = data_processed[(data_processed.index >= time_begin) & (data_processed.index < time_end)]

            if len(hourly_data) > 0:
                ax_right.plot(idx_b2, hourly_data.iloc[0], '.', ms=4, c=colors[i], alpha=0.5)

# ===== LHCb =====
# Left plot: Mean luminosity evolution
ax_left = ax[2, 0]
ax_left.set_xlabel("Time", fontsize=12)
ax_left.set_ylabel(r"$\mathcal{L}$ (10$^{35}$ m$^{-2}$ s$^{-1}$)", fontsize=12)
ax_left.set_title("LHCb - Mean Luminosity", fontsize=13, fontweight='bold')

if 'LHCB:BUNCH_LUMI_INST' in df.columns:
    data_lhcb = df['LHCB:BUNCH_LUMI_INST'].dropna()
    if len(data_lhcb) > 0:
        # Calculate mean and std over colliding bunches (filter out zeros)
        mean_lumi = data_lhcb.apply(lambda x: np.mean(x['elements'][idx_b2_coll][x['elements'][idx_b2_coll] != 0]) if isinstance(x, dict) and 'elements' in x else np.nan)
        std_lumi = data_lhcb.apply(lambda x: np.std(x['elements'][idx_b2_coll][x['elements'][idx_b2_coll] != 0]) if isinstance(x, dict) and 'elements' in x else np.nan)

        # Convert from Hz/ub to 10^35 m^-2 s^-1
        mean_lumi = mean_lumi * 1e-1
        std_lumi = std_lumi * 1e-1

        ax_left.fill_between(mean_lumi.index, mean_lumi - std_lumi, mean_lumi + std_lumi,
                            color="g", alpha=0.3)
        ax_left.plot(mean_lumi, c="g", lw=2, label="LHCb")
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(fontsize=14)

ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax_left.xaxis.set_major_locator(MaxNLocator(5))
plt.setp(ax_left.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Right plot: Bunch-by-bunch snapshots (one per hour)
ax_right = ax[2, 1]
ax_right.set_xlabel("Bunch slot (25 ns)", fontsize=12)
ax_right.set_title("LHCb - BBB Luminosity (one snapshot/hour)", fontsize=13, fontweight='bold')
ax_right.grid(True, alpha=0.3)

if 'LHCB:BUNCH_LUMI_INST' in df.columns:
    data_lhcb = df['LHCB:BUNCH_LUMI_INST'].dropna()
    if len(data_lhcb) > 0:
        # Extract elements and convert to 10^35 m^-2 s^-1
        data_processed = data_lhcb.apply(lambda x: x['elements'][idx_b2] * 1e-1 if isinstance(x, dict) and 'elements' in x else np.nan)

        total_hours = int((data_processed.index[-1] - data_processed.index[0]).total_seconds() // 3600)
        colors = cm.rainbow(np.linspace(0, 1, total_hours + 1))
        start_time = data_processed.index[0]

        # Plot one snapshot per hour
        for i in range(0, total_hours + 1, every_hours):
            time_begin = start_time + pd.Timedelta(hours=i)
            time_end = time_begin + pd.Timedelta(hours=1)
            hourly_data = data_processed[(data_processed.index >= time_begin) & (data_processed.index < time_end)]

            if len(hourly_data) > 0:
                ax_right.plot(idx_b2, hourly_data.iloc[0], '.', ms=4, c=colors[i], alpha=0.5)

plt.tight_layout()
plt.show()

# %%
# Print summary statistics
print("\n" + "="*80)
print(f"Luminosity Summary for Fill {fill_number}")
print("="*80)

for exp in ['ATLAS', 'CMS', 'LHCB']:
    var_name = f'{exp}:BUNCH_LUMI_INST'
    if var_name in df.columns:
        data = df[var_name].dropna()
        if len(data) > 0:
            # Use appropriate colliding bunches
            idx_coll = idx_b1_coll if exp == 'ATLAS' else idx_b2_coll

            # Calculate mean over colliding bunches (filter zeros)
            mean_series = data.apply(lambda x: np.mean(x['elements'][idx_coll][x['elements'][idx_coll] != 0]) if isinstance(x, dict) and 'elements' in x else np.nan)

            # Convert to 10^35 m^-2 s^-1
            mean_series = mean_series * 1e-1

            peak = mean_series.max()
            mean = mean_series.mean()
            final = mean_series.iloc[-1]

            print(f"\n  {exp}:")
            print(f"    Peak:  {peak:.2f} x 10^35 m^-2 s^-1")
            print(f"    Mean:  {mean:.2f} x 10^35 m^-2 s^-1")
            print(f"    Final: {final:.2f} x 10^35 m^-2 s^-1")

print("\nDone!")

# %%
