"""
Simple example: Retrieve and plot intensity and crossing angles for a given fill.

This script demonstrates how to:
1. Get data for a specific fill from NXCALS
2. Retrieve beam intensity and crossing angles at IP1 and IP5
3. Plot the data versus time
"""

# %%
# Setup: Import packages and create spark session
import nx2pd as nx
import pandas as pd
import matplotlib.pyplot as plt
from nxcals.spark_session_builder import get_or_create, Flavor
import matplotlib.dates as mdates

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

# Variable names:
# - Beam intensity for both beams
# - Crossing angles at all main IPs (both H and V) - using LhcStateTracker
variables = [
    'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY',                    # Beam 1 intensity
    'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY',                    # Beam 2 intensity

    # Crossing angles at main IPs (both H and V for all IPs)
    'LhcStateTracker:LHCBEAM:IP1-XING-H-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP1-XING-V-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP2-XING-H-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP2-XING-V-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP5-XING-H-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP5-XING-V-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP8-XING-H-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP8-XING-V-MURAD:value',
]

# %%
# Get the fill data
print(f"Retrieving data for fill {fill_number}...")
df = sk.get_fill_raw_data(fill_number, variables)
print(f"Retrieved {len(df)} data points.")
print("\nDataFrame info:")
print(df.head())

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

# %%
# Plot the data
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Format the fill start date and duration for titles
fill_date_str = fill_start.strftime('%Y-%m-%d')

# Update title based on whether we're analyzing full fill or a subset
if start_mode is not None or end_mode is not None:
    mode_range = f"{start_mode or 'START'} → {end_mode or 'END'}"
    title_info = f'Fill {fill_number} | {fill_date_str} | {mode_range} | Duration: {analysis_duration_hours:.2f}h'
else:
    title_info = f'Fill {fill_number} | {fill_date_str} | Duration: {analysis_duration_hours:.2f}h'

# Plot 1: Beam intensities
ax1 = axes[0]
if 'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY' in df.columns:
    data = df['LHC.BCTDC.A6R4.B1:BEAM_INTENSITY'].dropna()
    ax1.plot(data.index, data.values, label='Beam 1', linewidth=1.5)
if 'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY' in df.columns:
    data = df['LHC.BCTDC.A6R4.B2:BEAM_INTENSITY'].dropna()
    ax1.plot(data.index, data.values, label='Beam 2', linewidth=1.5)
ax1.set_ylabel('Beam Intensity')
ax1.set_xlabel('Time (HH:MM)')
ax1.set_title(f'{title_info}\nBeam Intensity vs Time')
ax1.legend()
ax1.grid(True, alpha=0.3)
# Format x-axis to show HH:MM
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Crossing angles (all IPs, both H and V)
ax2 = axes[1]

# Define crossing angle variables with formatting
xing_vars = [
    ('LhcStateTracker:LHCBEAM:IP1-XING-H-MURAD:value', 'IP1 (H)', 'blue', 'o', '--'),
    ('LhcStateTracker:LHCBEAM:IP1-XING-V-MURAD:value', 'IP1 (V)', 'lightblue', 'o', '-'),
    ('LhcStateTracker:LHCBEAM:IP2-XING-H-MURAD:value', 'IP2 (H)', 'orange', 'o', '--'),
    ('LhcStateTracker:LHCBEAM:IP2-XING-V-MURAD:value', 'IP2 (V)', 'lightyellow', 'o', '-'),
    ('LhcStateTracker:LHCBEAM:IP5-XING-H-MURAD:value', 'IP5 (H)', 'coral', 'o', '--'),
    ('LhcStateTracker:LHCBEAM:IP5-XING-V-MURAD:value', 'IP5 (V)', 'red', 'o', '-'),
    ('LhcStateTracker:LHCBEAM:IP8-XING-H-MURAD:value', 'IP8 (H)', 'lightgreen', 'o', '--'),
    ('LhcStateTracker:LHCBEAM:IP8-XING-V-MURAD:value', 'IP8 (V)', 'green', 'o', '-'),
]

for var_name, label, color, marker, linestyle in xing_vars:
    if var_name in df.columns:
        data = df[var_name].dropna()
        if len(data) > 0:
            # Mask zero values (angles not used)
            data_nonzero = data[data.abs() > 0.1]  # Use small threshold to account for numerical noise
            if len(data_nonzero) > 0:
                ax2.plot(data_nonzero.index, data_nonzero.values, color=color, marker=marker,
                        markersize=3, linewidth=1.5, label=label, alpha=0.8, linestyle=linestyle)
                # Print final non-zero value
                final_val = data_nonzero.iloc[-1]
                print(f"  {label}: {final_val:.1f} μrad")

ax2.set_ylabel('Crossing Angle [μrad]', fontsize=11)
ax2.set_xlabel('Time (HH:MM)', fontsize=11)
ax2.set_title(f'Crossing Angles at Main Interaction Points', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9, ncol=2)
ax2.grid(True, alpha=0.3)
# Format x-axis to show HH:MM
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# %%
# Print summary statistics
print("\n" + "="*60)
print(f"Summary for Fill {fill_number}")
print("="*60)
for var in variables:
    if var in df.columns:
        data = df[var].dropna()
        if len(data) > 0:
            print(f"\n{var}:")
            print(f"  Mean:   {data.mean():.2f}")
            print(f"  Min:    {data.min():.2f}")
            print(f"  Max:    {data.max():.2f}")
            print(f"  Points: {len(data)}")

print("\nDone!")
# %%
