"""
Luminosity analysis script: Visualize total instantaneous luminosity during a fill.

This script demonstrates how to:
1. Retrieve total instantaneous luminosity for ATLAS, CMS, LHCb, and ALICE
2. Plot luminosity evolution over time (overlay and individual plots)
"""

# %%
# Setup: Import packages and create spark session
import nx2pd as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from nxcals.spark_session_builder import get_or_create, Flavor

# Create spark session (using LOCAL flavor for simplicity)
print("Creating Spark session...")
spark = get_or_create(flavor=Flavor.LOCAL)
sk = nx.SparkIt(spark)
print("Spark session created.")

# %%
# Define the fill number and variables to retrieve
fill_number = 10993

# Total instantaneous luminosity for all experiments
# Units: Hz/ub (Hz per microbarn)
# Conversion: 1 Hz/ub = 10^30 cm^-2 s^-1 = 10^-4 * 10^34 cm^-2 s^-1
total_lumi_variables = [
    'ATLAS:LUMI_TOT_INST',   # Hz/ub
    'CMS:LUMI_TOT_INST',     # Hz/ub
    'LHCB:LUMI_TOT_INST',    # Hz/ub
    'ALICE:LUMI_TOT_INST',   # Hz/ub
]

# %%
# Get the fill data for total luminosity
print(f"Retrieving total luminosity data for fill {fill_number}...")
df_total = sk.get_fill_raw_data(fill_number, total_lumi_variables)
print(f"Retrieved {len(df_total)} data points for total luminosity.")

# Get fill timing information
fill_info = sk.get_fill_time(fill_number)
fill_start = fill_info['start']
fill_duration_hours = fill_info['duration'].total_seconds() / 3600

# Check which experiments have data
experiments_with_data = []
for var in total_lumi_variables:
    if var in df_total.columns:
        data = df_total[var].dropna()
        if len(data) > 0:
            exp_name = var.split(':')[0]
            experiments_with_data.append(exp_name)
            print(f"  ✓ {exp_name}: {len(data)} data points")
        else:
            print(f"  ✗ {var}: No data")
    else:
        print(f"  ✗ {var}: Variable not found")

# %%
# PLOT 1: Overlay plot - all experiments on same axes
fig1, ax = plt.subplots(figsize=(14, 8))

# Format the fill start date and duration for title
fill_date_str = fill_start.strftime('%Y-%m-%d')
title_info = f'Fill {fill_number} | {fill_date_str} | Duration: {fill_duration_hours:.2f}h'

# Plot each experiment with different colors
experiments = ['ATLAS', 'CMS', 'LHCB', 'ALICE']
colors = ['blue', 'red', 'green', 'orange']
linestyles = ['-', '-', '--', '--']

for exp, color, ls in zip(experiments, colors, linestyles):
    var_name = f'{exp}:LUMI_TOT_INST'

    if var_name in df_total.columns:
        data = df_total[var_name].dropna()
        if len(data) > 0:
            # Convert from Hz/ub to 10^34 cm^-2 s^-1
            lumi_converted = data.values * 1e-4
            ax.plot(data.index, lumi_converted, color=color, linewidth=2,
                   linestyle=ls, label=exp, alpha=0.8)

ax.set_ylabel('Luminosity [10³⁴ cm⁻² s⁻¹]', fontsize=12)
ax.set_xlabel('Time (HH:MM)', fontsize=12)
ax.set_title(f'{title_info}\nTotal Instantaneous Luminosity - All Experiments',
            fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# %%
# PLOT 2: Individual plots for each experiment (2x2 grid)
fig2, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
fig2.suptitle(f'{title_info}\nTotal Instantaneous Luminosity - Individual Experiments',
             fontsize=14, fontweight='bold')

# Flatten axes for easier iteration
axes_flat = axes.flatten()

for i, (exp, color) in enumerate(zip(experiments, colors)):
    ax = axes_flat[i]
    var_name = f'{exp}:LUMI_TOT_INST'

    if var_name in df_total.columns:
        data = df_total[var_name].dropna()
        if len(data) > 0:
            # Convert from Hz/ub to 10^34 cm^-2 s^-1
            lumi_converted = data.values * 1e-4
            ax.plot(data.index, lumi_converted, color=color, linewidth=1.5)
            ax.set_ylabel('Lumi [10³⁴ cm⁻² s⁻¹]', fontsize=10)
            ax.set_title(f'{exp}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add statistics as text
            mean_lumi = lumi_converted.mean()
            max_lumi = lumi_converted.max()
            ax.text(0.02, 0.98, f'Peak: {max_lumi:.2f}\nMean: {mean_lumi:.2f}',
                   transform=ax.transAxes, verticalalignment='top',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No data available',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='gray')
            ax.set_title(f'{exp}', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Variable not found',
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='gray')
        ax.set_title(f'{exp}', fontsize=12, fontweight='bold')

    # Format x-axis for bottom row
    if i >= 2:
        ax.set_xlabel('Time (HH:MM)', fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# %%
# Print summary
print("\n" + "="*80)
print(f"Luminosity Summary for Fill {fill_number}")
print("="*80)

if len(experiments_with_data) > 0:
    print("\nExperiments with total luminosity data:")
    for exp in experiments_with_data:
        var_name = f'{exp}:LUMI_TOT_INST'
        if var_name in df_total.columns:
            data = df_total[var_name].dropna()
            if len(data) > 0:
                # Convert to 10^34 cm^-2 s^-1
                lumi_converted = data.values * 1e-4
                peak = lumi_converted.max()
                mean = lumi_converted.mean()
                final = lumi_converted[-1]
                print(f"\n  {exp}:")
                print(f"    Peak:  {peak:.2f} × 10³⁴ cm⁻² s⁻¹")
                print(f"    Mean:  {mean:.2f} × 10³⁴ cm⁻² s⁻¹")
                print(f"    Final: {final:.2f} × 10³⁴ cm⁻² s⁻¹")
else:
    print("\nNo luminosity data available for this fill.")

print("\nDone!")

# %%
