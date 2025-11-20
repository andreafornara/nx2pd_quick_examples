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

# Create spark session (using LOCAL flavor for simplicity)
print("Creating Spark session...")
spark = get_or_create(flavor=Flavor.LOCAL)
sk = nx.SparkIt(spark)
print("Spark session created.")

# %%
# Define the fill number and variables to retrieve
fill_number = 10993

# Variable names:
# - Beam intensity for both beams
# - Crossing angles at IP1 (vertical) and IP5 (horizontal)
variables = [
    'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY',  # Beam 1 intensity
    'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY',  # Beam 2 intensity
    'LHC.RUNCONFIG:IP1-XING-V-MURAD',    # IP1 crossing angle (vertical)
    'LHC.RUNCONFIG:IP5-XING-H-MURAD',    # IP5 crossing angle (horizontal)
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
fill_start = fill_info['start']
fill_duration_hours = fill_info['duration'].total_seconds() / 3600

# %%
# Plot the data
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Format the fill start date and duration for titles
fill_date_str = fill_start.strftime('%Y-%m-%d')
title_info = f'Fill {fill_number} | {fill_date_str} | Duration: {fill_duration_hours:.2f}h'

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

# Plot 2: Crossing angles
ax2 = axes[1]
if 'LHC.RUNCONFIG:IP1-XING-V-MURAD' in df.columns:
    data = df['LHC.RUNCONFIG:IP1-XING-V-MURAD'].dropna()
    ax2.plot(data.index, data.values, label='IP1 (vertical)',
             linewidth=1.5, marker='o', markersize=3)
if 'LHC.RUNCONFIG:IP5-XING-H-MURAD' in df.columns:
    data = df['LHC.RUNCONFIG:IP5-XING-H-MURAD'].dropna()
    ax2.plot(data.index, data.values, label='IP5 (horizontal)',
             linewidth=1.5, marker='o', markersize=3)
ax2.set_ylabel('Crossing Angle (μrad)')
ax2.set_xlabel('Time (HH:MM)')
ax2.set_title(f'Crossing Angles vs Time')
ax2.legend()
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
