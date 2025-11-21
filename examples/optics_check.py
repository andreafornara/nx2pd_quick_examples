"""
Optics check script: Visualize optics changes during a fill.

This script demonstrates how to:
1. Retrieve optics and beam data for a specific fill from NXCALS
2. Plot beam energy and intensity with optics as colored background
3. Create a dedicated optics timeline showing transitions clearly
"""

# %%
# Setup: Import packages and create spark session
import nx2pd as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from nxcals.spark_session_builder import get_or_create, Flavor

# Import configuration and helper functions
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from useful_functions import plot_time_windows_background

# Create spark session (using LOCAL flavor for simplicity)
print("Creating Spark session...")
spark = get_or_create(flavor=Flavor.LOCAL)
sk = nx.SparkIt(spark)
print("Spark session created.")

# %%
# Define the fill number and variables to retrieve
# These values are imported from config.py
fill_number = FILL_NUMBER
start_mode = START_MODE
end_mode = END_MODE

# Variable names for optics and beam parameters
variables = [
    'LHC.BCCM.B1.A:BEAM_ENERGY',           # Beam energy
    'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY',    # Beam 1 intensity
    'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY',    # Beam 2 intensity
    'LhcStateTracker:State:opticName',     # Optics name
    'LhcStateTracker:State:beamProcess',   # Beam process
    'HX:BMODE',                             # Beam mode
]

# %%
# Get the fill data
print(f"Retrieving data for fill {fill_number}...")
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

# Display unique optics used during the fill
if 'LhcStateTracker:State:opticName' in df.columns:
    optics_used = df['LhcStateTracker:State:opticName'].dropna().unique()
    print(f"\nOptics used during fill {fill_number}:")
    for optics in optics_used:
        print(f"  - {optics}")

# %%
# PLOT 1: Energy and Intensity with colored background
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Format the fill start date and duration for title
fill_date_str = fill_start.strftime('%Y-%m-%d')

# Update title based on whether we're analyzing full fill or a subset
if start_mode is not None or end_mode is not None:
    mode_range = f"{start_mode or 'START'} → {end_mode or 'END'}"
    title_info = f'Fill {fill_number} | {fill_date_str} | {mode_range} | Duration: {analysis_duration_hours:.2f}h'
else:
    title_info = f'Fill {fill_number} | {fill_date_str} | Duration: {analysis_duration_hours:.2f}h'

# Subplot 1: Beam Energy
if 'LHC.BCCM.B1.A:BEAM_ENERGY' in df.columns:
    data = df['LHC.BCCM.B1.A:BEAM_ENERGY'].dropna()
    # Convert to TeV
    ax1.plot(data.index, data.values / 1000, 'k-', linewidth=2, label='Beam Energy')
ax1.set_ylabel('Energy [TeV]', fontsize=11)
ax1.set_title(f'{title_info}\nBeam Energy vs Time', fontsize=12, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 7)

# Add optics background to energy plot
colormap = None
if 'LhcStateTracker:State:opticName' in df.columns:
    colormap = plot_time_windows_background(ax1, df['LhcStateTracker:State:opticName'].dropna())

# Subplot 2: Beam Intensities
if 'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY' in df.columns:
    data = df['LHC.BCTDC.A6R4.B1:BEAM_INTENSITY'].dropna()
    ax2.plot(data.index, data.values, 'b-', linewidth=1.5, label='Beam 1')
if 'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY' in df.columns:
    data = df['LHC.BCTDC.A6R4.B2:BEAM_INTENSITY'].dropna()
    ax2.plot(data.index, data.values, 'r-', linewidth=1.5, label='Beam 2')
ax2.set_ylabel('Beam Intensity', fontsize=11)
ax2.set_xlabel('Time (HH:MM)', fontsize=11)
ax2.set_title('Beam Intensity vs Time', fontsize=12)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Add optics background to intensity plot
if 'LhcStateTracker:State:opticName' in df.columns and colormap is not None:
    plot_time_windows_background(ax2, df['LhcStateTracker:State:opticName'].dropna(),
                                 colormap=colormap)

# Format x-axis - fixed to show proper time
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'fill_{fill_number}_optics_energy_intensity.png', dpi=150, bbox_inches='tight')
print(f"\nPlot 1 saved as 'fill_{fill_number}_optics_energy_intensity.png'")
plt.show()

# %%
# Print summary
print("\n" + "="*80)
print(f"Optics Summary for Fill {fill_number}")
print("="*80)

if 'LhcStateTracker:State:opticName' in df.columns:
    optics_series = df['LhcStateTracker:State:opticName'].dropna()

    # Find transitions
    value_changes = optics_series != optics_series.shift(1)
    change_indices = value_changes[value_changes].index

    print("\nOptics Transitions:")
    for i in range(len(change_indices)):
        change_time = change_indices[i]
        optics = optics_series[change_time]
        time_str = change_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {time_str} → {optics}")

print("\nDone!")

# %%