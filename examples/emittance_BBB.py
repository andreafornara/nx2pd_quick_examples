"""
Bunch-by-Bunch Emittance Analysis

This script performs bunch-by-bunch analysis of BSRT emittance measurements
for both beams during different phases of the fill.

Requirements:
- Large Spark configuration due to high data volume
- BSRT emittance data for both beams (horizontal and vertical)
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

# Import configuration
import sys
import os
import yaml

# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

FILL_NUMBER = config['fill_number']
START_MODE = config['start_mode']
END_MODE = config['end_mode']
SNAPSHOT_INTERVAL_HOURS = config['snapshot_interval_hours']
SPARK_LARGE_MEMORY_CONFIG = config['spark_large_memory']

# Create spark session with large memory configuration for bunch-by-bunch data
print("Creating Spark session with large memory configuration...")
spark = get_or_create(flavor=Flavor.LOCAL, conf=SPARK_LARGE_MEMORY_CONFIG)
sk = nx.SparkIt(spark)
print("Spark session created.")

# %%
# Define the fill number and analysis parameters
# These values are imported from config.py
fill_number = FILL_NUMBER
start_mode = START_MODE
end_mode = END_MODE
every_hours = SNAPSHOT_INTERVAL_HOURS

# Variables for bunch-by-bunch emittance analysis
variables = [
    # BSRT emittance measurements (bunch-by-bunch)
    'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_H',
    'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_V',
    'LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_H',
    'LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_V',

    # Additional context variables
    'LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY',
    'LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY',
    'HX:BETASTAR_IP1',
]

# %%
# Get the fill data
print(f"\nRetrieving bunch-by-bunch data for fill {fill_number}...")
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
                print(f"\n>>> Analysis START configured: {start_mode} mode")
                print(f"    Start time: {analysis_start}")
                break
        else:
            print(f"\n>>> WARNING: Start mode '{start_mode}' not found, using fill start")
            print(f"    Start time: {analysis_start}")

    # Find end time
    if end_mode is not None:
        for mode_info in modes:
            if mode_info['mode'] == end_mode:
                analysis_end = mode_info['end']
                print(f">>> Analysis END configured: {end_mode} mode")
                print(f"    End time: {analysis_end}")
                break
        else:
            print(f">>> WARNING: End mode '{end_mode}' not found, using fill end")
            print(f"    End time: {analysis_end}")

# Filter dataframe to analysis window
df = df[(df.index >= analysis_start) & (df.index <= analysis_end)]
print(f"Filtered to analysis window: {len(df)} data points.")

# Calculate duration and create title info
fill_start = fill_info['start']
analysis_duration_hours = (analysis_end - analysis_start).total_seconds() / 3600
fill_date_str = fill_start.strftime('%Y-%m-%d')

# Update title based on whether we're analyzing full fill or a subset
if start_mode is not None or end_mode is not None:
    mode_range = f"{start_mode or 'START'} to {end_mode or 'END'}"
    title_info = f'Fill {fill_number} | {fill_date_str} | {mode_range} | Duration: {analysis_duration_hours:.2f}h'
else:
    title_info = f'Fill {fill_number} | {fill_date_str} | Duration: {analysis_duration_hours:.2f}h'

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

    print(f"Beam 1: {len(idx_b1)} filled bunches")
    print(f"Beam 2: {len(idx_b2)} filled bunches")
except Exception as e:
    print(f"WARNING: Could not determine filling scheme: {e}")
    print(f"Error details: {type(e).__name__}")
    idx_b1 = np.arange(0, 100)
    idx_b2 = np.arange(0, 100)

# %%
# PLOT 1: Mean emittance evolution (averaged over all filled bunches)
print("\n" + "="*80)
print("PLOT 1: Mean Emittance Evolution")
print("="*80)

fig1, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
fig1.suptitle(f'{title_info}\nMean Bunch Emittance (BSRT)', fontsize=14, fontweight='bold')

# Beam 1 Horizontal
ax = axes[0, 0]
if 'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_H' in df.columns:
    data = df['LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_H'].dropna()
    if len(data) > 0:
        mean_emit = data.apply(lambda x: np.mean(x['elements'][idx_b1]))
        ax.plot(mean_emit.index, mean_emit.values, 'b-', linewidth=2, label='Mean')
        # Also plot std as shaded area
        std_emit = data.apply(lambda x: np.std(x['elements'][idx_b1]))
        ax.fill_between(mean_emit.index, mean_emit - std_emit, mean_emit + std_emit,
                        alpha=0.3, color='b', label=r'$\pm 1\sigma$')
        print(f"  B1H: Mean = {mean_emit.iloc[0]:.2f} -> {mean_emit.iloc[-1]:.2f} um")

ax.set_ylabel(r'Emittance [$\mu$m]', fontsize=11)
ax.set_title('Beam 1 - Horizontal', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Beam 1 Vertical
ax = axes[0, 1]
if 'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_V' in df.columns:
    data = df['LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_V'].dropna()
    if len(data) > 0:
        mean_emit = data.apply(lambda x: np.mean(x['elements'][idx_b1]))
        ax.plot(mean_emit.index, mean_emit.values, 'r-', linewidth=2, label='Mean')
        std_emit = data.apply(lambda x: np.std(x['elements'][idx_b1]))
        ax.fill_between(mean_emit.index, mean_emit - std_emit, mean_emit + std_emit,
                        alpha=0.3, color='r', label=r'$\pm 1\sigma$')
        print(f"  B1V: Mean = {mean_emit.iloc[0]:.2f} -> {mean_emit.iloc[-1]:.2f} um")

ax.set_ylabel(r'Emittance [$\mu$m]', fontsize=11)
ax.set_title('Beam 1 - Vertical', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Beam 2 Horizontal
ax = axes[1, 0]
if 'LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_H' in df.columns:
    data = df['LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_H'].dropna()
    if len(data) > 0:
        mean_emit = data.apply(lambda x: np.mean(x['elements'][idx_b2]))
        ax.plot(mean_emit.index, mean_emit.values, 'b-', linewidth=2, label='Mean')
        std_emit = data.apply(lambda x: np.std(x['elements'][idx_b2]))
        ax.fill_between(mean_emit.index, mean_emit - std_emit, mean_emit + std_emit,
                        alpha=0.3, color='b', label=r'$\pm 1\sigma$')
        print(f"  B2H: Mean = {mean_emit.iloc[0]:.2f} -> {mean_emit.iloc[-1]:.2f} um")

ax.set_ylabel(r'Emittance [$\mu$m]', fontsize=11)
ax.set_xlabel('Time (HH:MM)', fontsize=11)
ax.set_title('Beam 2 - Horizontal', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Beam 2 Vertical
ax = axes[1, 1]
if 'LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_V' in df.columns:
    data = df['LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_V'].dropna()
    if len(data) > 0:
        mean_emit = data.apply(lambda x: np.mean(x['elements'][idx_b2]))
        ax.plot(mean_emit.index, mean_emit.values, 'r-', linewidth=2, label='Mean')
        std_emit = data.apply(lambda x: np.std(x['elements'][idx_b2]))
        ax.fill_between(mean_emit.index, mean_emit - std_emit, mean_emit + std_emit,
                        alpha=0.3, color='r', label=r'$\pm 1\sigma$')
        print(f"  B2V: Mean = {mean_emit.iloc[0]:.2f} -> {mean_emit.iloc[-1]:.2f} um")

ax.set_ylabel(r'Emittance [$\mu$m]', fontsize=11)
ax.set_xlabel('Time (HH:MM)', fontsize=11)
ax.set_title('Beam 2 - Vertical', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# %%
# PLOT 2: Bunch-by-bunch emittance with mean evolution and hourly snapshots
print("\n" + "="*80)
print("PLOT 2: Bunch-by-Bunch Emittance (Mean + BBB Snapshots)")
print("="*80)

# Loop over both beams
for flag, beam in zip(["R", "L"], [1, 2]):
    idx_beam = idx_b1 if beam == 1 else idx_b2
    idx_coll = idx_b1  # For simplicity, using all filled bunches as colliding
    idx_noncoll = np.array([])  # Empty for now, can be implemented if needed

    fig, ax = plt.subplots(figsize=(18, 16), nrows=2, ncols=2,
                           gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [0.5, 1]},
                           sharey=True)
    plt.suptitle(f'{title_info}\nBeam {beam} Emittance', fontsize=20, fontweight='bold')

    # ===== HORIZONTAL EMITTANCE =====
    # Left plot: Mean emittance evolution
    ax_left = ax[0, 0]
    ax_left.set_xlabel("Time", fontsize=12)
    ax_left.set_ylabel(r"Emittance B$" + str(beam) + r"$H [$\mu$m]", fontsize=12)
    ax_left.set_title(f"Beam {beam} - Horizontal (Mean)", fontsize=13, fontweight='bold')

    var_h = f'LHC.BSRT.5{flag}4.B{beam}:BUNCH_EMITTANCE_H'
    if var_h in df.columns:
        data_h = df[var_h].dropna()
        if len(data_h) > 0:
            # Calculate mean and std over colliding bunches (filter out zeros)
            mean_emit = data_h.apply(lambda x: np.mean(x['elements'][idx_coll][x['elements'][idx_coll] != 0]) if isinstance(x, dict) and 'elements' in x else np.nan)
            std_emit = data_h.apply(lambda x: np.std(x['elements'][idx_coll][x['elements'][idx_coll] != 0]) if isinstance(x, dict) and 'elements' in x else np.nan)

            ax_left.fill_between(mean_emit.index, mean_emit - std_emit, mean_emit + std_emit,
                                color="k", alpha=0.3)
            ax_left.plot(mean_emit, c="k", lw=2)
            ax_left.grid(True, alpha=0.3)

    ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_left.xaxis.set_major_locator(MaxNLocator(5))
    plt.setp(ax_left.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Right plot: Bunch-by-bunch snapshots (one per hour)
    ax_right = ax[0, 1]
    ax_right.set_xlabel("Bunch slot (25 ns)", fontsize=12)
    ax_right.set_title(f"Beam {beam} - Horizontal BBB (one snapshot/hour)", fontsize=13, fontweight='bold')
    ax_right.grid(True, alpha=0.3)

    if var_h in df.columns:
        data_h = df[var_h].dropna()
        if len(data_h) > 0:
            # Extract elements
            data_processed = data_h.apply(lambda x: x['elements'][idx_beam] if isinstance(x, dict) and 'elements' in x else np.nan)

            total_hours = int((data_processed.index[-1] - data_processed.index[0]).total_seconds() // 3600)
            colors = cm.rainbow(np.linspace(0, 1, total_hours + 1))
            start_time = data_processed.index[0]

            # Plot one snapshot per hour
            for i in range(0, total_hours + 1, every_hours):
                time_begin = start_time + pd.Timedelta(hours=i)
                time_end = time_begin + pd.Timedelta(hours=1)
                hourly_data = data_processed[(data_processed.index >= time_begin) & (data_processed.index < time_end)]

                if len(hourly_data) > 0:
                    ax_right.plot(idx_beam, hourly_data.iloc[0], '.', ms=4, c=colors[i], alpha=0.5)

    # ===== VERTICAL EMITTANCE =====
    # Left plot: Mean emittance evolution
    ax_left = ax[1, 0]
    ax_left.set_xlabel("Time", fontsize=12)
    ax_left.set_ylabel(r"Emittance B$" + str(beam) + r"$V [$\mu$m]", fontsize=12)
    ax_left.set_title(f"Beam {beam} - Vertical (Mean)", fontsize=13, fontweight='bold')

    var_v = f'LHC.BSRT.5{flag}4.B{beam}:BUNCH_EMITTANCE_V'
    if var_v in df.columns:
        data_v = df[var_v].dropna()
        if len(data_v) > 0:
            # Calculate mean and std over colliding bunches (filter out zeros)
            mean_emit = data_v.apply(lambda x: np.mean(x['elements'][idx_coll][x['elements'][idx_coll] != 0]) if isinstance(x, dict) and 'elements' in x else np.nan)
            std_emit = data_v.apply(lambda x: np.std(x['elements'][idx_coll][x['elements'][idx_coll] != 0]) if isinstance(x, dict) and 'elements' in x else np.nan)

            ax_left.fill_between(mean_emit.index, mean_emit - std_emit, mean_emit + std_emit,
                                color="k", alpha=0.3)
            ax_left.plot(mean_emit, c="k", lw=2)
            ax_left.grid(True, alpha=0.3)

    ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_left.xaxis.set_major_locator(MaxNLocator(5))
    plt.setp(ax_left.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Right plot: Bunch-by-bunch snapshots (one per hour)
    ax_right = ax[1, 1]
    ax_right.set_xlabel("Bunch slot (25 ns)", fontsize=12)
    ax_right.set_title(f"Beam {beam} - Vertical BBB (one snapshot/hour)", fontsize=13, fontweight='bold')
    ax_right.grid(True, alpha=0.3)

    if var_v in df.columns:
        data_v = df[var_v].dropna()
        if len(data_v) > 0:
            # Extract elements
            data_processed = data_v.apply(lambda x: x['elements'][idx_beam] if isinstance(x, dict) and 'elements' in x else np.nan)

            total_hours = int((data_processed.index[-1] - data_processed.index[0]).total_seconds() // 3600)
            colors = cm.rainbow(np.linspace(0, 1, total_hours + 1))
            start_time = data_processed.index[0]

            # Plot one snapshot per hour
            for i in range(0, total_hours + 1, every_hours):
                time_begin = start_time + pd.Timedelta(hours=i)
                time_end = time_begin + pd.Timedelta(hours=1)
                hourly_data = data_processed[(data_processed.index >= time_begin) & (data_processed.index < time_end)]

                if len(hourly_data) > 0:
                    ax_right.plot(idx_beam, hourly_data.iloc[0], '.', ms=4, c=colors[i], alpha=0.5)

    plt.tight_layout()
    plt.show()

# %%
# PLOT 3: Heatmap of emittance evolution for Beam 1
print("\n" + "="*80)
print("PLOT 3: Beam 1 Emittance Evolution Heatmap")
print("="*80)

fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
fig3.suptitle(f'{title_info}\nBeam 1 Emittance Evolution (Bunch-by-Bunch)',
             fontsize=14, fontweight='bold')

# Beam 1 Horizontal
if 'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_H' in df.columns:
    data = df['LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_H'].dropna()
    if len(data) > 0:
        # Create 2D array: time x bunch slot
        emit_matrix = np.array([row['elements'][idx_b1] for row in data.values])

        # Plot heatmap
        im1 = ax1.imshow(emit_matrix.T, aspect='auto', origin='lower',
                        cmap='viridis', interpolation='nearest',
                        extent=[0, len(data), idx_b1[0], idx_b1[-1]])
        ax1.set_ylabel('Bunch Slot', fontsize=11)
        ax1.set_title('Horizontal Emittance', fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label(r'Emittance [$\mu$m]', fontsize=10)

# Beam 1 Vertical
if 'LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_V' in df.columns:
    data = df['LHC.BSRT.5R4.B1:BUNCH_EMITTANCE_V'].dropna()
    if len(data) > 0:
        # Create 2D array: time x bunch slot
        emit_matrix = np.array([row['elements'][idx_b1] for row in data.values])

        # Plot heatmap
        im2 = ax2.imshow(emit_matrix.T, aspect='auto', origin='lower',
                        cmap='viridis', interpolation='nearest',
                        extent=[0, len(data), idx_b1[0], idx_b1[-1]])
        ax2.set_ylabel('Bunch Slot', fontsize=11)
        ax2.set_xlabel('Time Index', fontsize=11)
        ax2.set_title('Vertical Emittance', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label(r'Emittance [$\mu$m]', fontsize=10)

plt.tight_layout()
plt.show()

# %%
# PLOT 4: Heatmap of emittance evolution for Beam 2
print("\n" + "="*80)
print("PLOT 4: Beam 2 Emittance Evolution Heatmap")
print("="*80)

fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
fig4.suptitle(f'{title_info}\nBeam 2 Emittance Evolution (Bunch-by-Bunch)',
             fontsize=14, fontweight='bold')

# Beam 2 Horizontal
if 'LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_H' in df.columns:
    data = df['LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_H'].dropna()
    if len(data) > 0:
        # Create 2D array: time x bunch slot
        emit_matrix = np.array([row['elements'][idx_b2] for row in data.values])

        # Plot heatmap
        im1 = ax1.imshow(emit_matrix.T, aspect='auto', origin='lower',
                        cmap='viridis', interpolation='nearest',
                        extent=[0, len(data), idx_b2[0], idx_b2[-1]])
        ax1.set_ylabel('Bunch Slot', fontsize=11)
        ax1.set_title('Horizontal Emittance', fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label(r'Emittance [$\mu$m]', fontsize=10)

# Beam 2 Vertical
if 'LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_V' in df.columns:
    data = df['LHC.BSRT.5L4.B2:BUNCH_EMITTANCE_V'].dropna()
    if len(data) > 0:
        # Create 2D array: time x bunch slot
        emit_matrix = np.array([row['elements'][idx_b2] for row in data.values])

        # Plot heatmap
        im2 = ax2.imshow(emit_matrix.T, aspect='auto', origin='lower',
                        cmap='viridis', interpolation='nearest',
                        extent=[0, len(data), idx_b2[0], idx_b2[-1]])
        ax2.set_ylabel('Bunch Slot', fontsize=11)
        ax2.set_xlabel('Time Index', fontsize=11)
        ax2.set_title('Vertical Emittance', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label(r'Emittance [$\mu$m]', fontsize=10)

plt.tight_layout()
plt.show()

# %%
# Print final summary
print("\n" + "="*80)
print(f"FILL {fill_number} - BUNCH-BY-BUNCH ANALYSIS COMPLETE")
print("="*80)
print(f"\nFill Information:")
print(f"  Fill start time: {fill_start}")
print(f"  Analysis window: {analysis_start} to {analysis_end}")
print(f"  Analysis duration: {analysis_duration_hours:.2f} hours")
print(f"\nFilling Scheme:")
print(f"  Beam 1: {len(idx_b1)} filled bunches")
print(f"  Beam 2: {len(idx_b2)} filled bunches")
print("\nDone!")

# %%
