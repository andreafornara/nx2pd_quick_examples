"""
Comprehensive Fill Analysis Script

This script provides a complete overview of a fill, showing:
1. Optics used during the fill
2. Crossing angles and beta* at main experiments (IP1, IP5, IP8)
3. Beam intensity and emittance
4. Luminosity at main experiments (ATLAS, CMS, LHCb, ALICE)
5. A summary plot with all information combined

Each metric is shown in a separate plot with proper titles and axis labels,
followed by a comprehensive summary plot.
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
# Define the fill number
fill_number = 10993

# Define time window for analysis (optional)
# Set to None to use full fill, or specify start/end modes from fill_info['modes']
# Examples: 'SETUP', 'INJPHYS', 'STABLE', 'BEAMDUMP', etc.
start_mode = 'SETUP' # None = use fill start, or specify mode like 'SETUP', 'INJPHYS', etc.
end_mode = 'BEAMDUMP'    # None = use fill end, or specify mode like 'STABLE', 'BEAMDUMP', etc.

# Define all variables needed for the analysis
variables = [
    # Optics information
    'LhcStateTracker:State:opticName',
    'HX:BETASTAR_IP1',
    'HX:BETASTAR_IP2',
    'HX:BETASTAR_IP5',
    'HX:BETASTAR_IP8',

    # Crossing angles at main IPs (new LhcStateTracker variables - both H and V for all IPs)
    'LhcStateTracker:LHCBEAM:IP1-XING-H-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP1-XING-V-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP2-XING-H-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP2-XING-V-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP5-XING-H-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP5-XING-V-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP8-XING-H-MURAD:value',
    'LhcStateTracker:LHCBEAM:IP8-XING-V-MURAD:value',

    # Beam intensity
    'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY',
    'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY',

    # Emittance (average from emittance scans)
    'EMITTANCE_SCAN_IP1:Emittance:HORIZONTAL_AverageEmittance_um',
    'EMITTANCE_SCAN_IP1:Emittance:VERTICAL_AverageEmittance_um',

    # Luminosity at main experiments
    'ATLAS:LUMI_TOT_INST',
    'CMS:LUMI_TOT_INST',
    'LHCB:LUMI_TOT_INST',
    'ALICE:LUMI_TOT_INST',
]

# %%
# Get the fill data
print(f"\nRetrieving data for fill {fill_number}...")
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
fill_date_str = fill_start.strftime('%Y-%m-%d')

# Update title based on whether we're analyzing full fill or a subset
if start_mode is not None or end_mode is not None:
    mode_range = f"{start_mode or 'START'} → {end_mode or 'END'}"
    title_info = f'Fill {fill_number} | {fill_date_str} | {mode_range} | Duration: {analysis_duration_hours:.2f}h'
else:
    title_info = f'Fill {fill_number} | {fill_date_str} | Duration: {analysis_duration_hours:.2f}h'

print(f"\nFill start: {fill_start}")
print(f"Analysis window start: {analysis_start}")
print(f"Analysis window end: {analysis_end}")
print(f"Analysis duration: {analysis_duration_hours:.2f} hours")

# %%
# Helper function for plotting optics background
def plot_time_windows_background(ax, df_column, colormap=None, alpha=0.3, zorder=-1):
    """
    Plot colored vertical bars as a background for time series plots.
    Each bar represents a time window where the value in df_column is constant.
    """
    # Drop NaN values
    series = df_column.dropna()

    if len(series) == 0:
        return {}

    # Get unique values and assign colors
    unique_values = series.unique()

    if colormap is None:
        # Create a default colormap
        cmap = plt.cm.get_cmap('tab10')
        colormap = {val: cmap(i / len(unique_values)) for i, val in enumerate(unique_values)}

    # Find time windows where the value changes
    value_changes = series != series.shift(1)
    change_indices = value_changes[value_changes].index

    # Create time windows
    windows = []
    for i in range(len(change_indices)):
        start_time = change_indices[i]
        end_time = change_indices[i + 1] if i + 1 < len(change_indices) else series.index[-1]
        value = series[start_time]
        windows.append((start_time, end_time, value))

    # Plot vertical bars for each window
    for start, end, value in windows:
        color = colormap.get(value, 'gray')
        ax.axvspan(start, end, alpha=alpha, color=color, zorder=zorder, label=value)

    return colormap

# %%
# PLOT 1: Optics Information
print("\n" + "="*80)
print("PLOT 1: Optics Information")
print("="*80)

fig1, ax1 = plt.subplots(figsize=(14, 6))

# Plot beta* for main IPs
ips = ['IP1', 'IP2', 'IP5', 'IP8']
colors_beta = ['blue', 'orange', 'red', 'green']
markers = ['o', 'v', 's', '^']

for ip, color, marker in zip(ips, colors_beta, markers):
    var_name = f'HX:BETASTAR_{ip}'
    if var_name in df.columns:
        data = df[var_name].dropna()
        if len(data) > 0:
            ax1.plot(data.index, data.values, color=color, marker=marker,
                    markersize=4, linewidth=2, label=f'β* {ip}', alpha=0.8)
            print(f"  {ip}: β* = {np.min(data[data>0]):.1f} cm (minimum β*)")

ax1.set_ylabel('β* [cm]', fontsize=12)
ax1.set_xlabel('Time (HH:MM)', fontsize=12)
ax1.set_title(f'{title_info}\nBeta* at Main Interaction Points', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add optics background coloring
optics_colormap = None
if 'LhcStateTracker:State:opticName' in df.columns:
    optics_used = df['LhcStateTracker:State:opticName'].dropna().unique()
    optics_colormap = plot_time_windows_background(ax1, df['LhcStateTracker:State:opticName'].dropna())
    print(f"\nOptics used during fill: {', '.join(optics_used)}")

plt.tight_layout()
plt.show()

# %%
# PLOT 2: Crossing Angles
print("\n" + "="*80)
print("PLOT 2: Crossing Angles at Main IPs")
print("="*80)
fig2, ax2 = plt.subplots(figsize=(18, 10))
xing_vars = [
    ('LhcStateTracker:LHCBEAM:IP1-XING-H-MURAD:value', 'IP1 (H)', 'blue', 'o', '--'),
    ('LhcStateTracker:LHCBEAM:IP1-XING-V-MURAD:value', 'IP1 (V)', 'lightblue', 'o', '--'),
    ('LhcStateTracker:LHCBEAM:IP2-XING-H-MURAD:value', 'IP2 (H)', 'orange', 'o', '--'),
    ('LhcStateTracker:LHCBEAM:IP2-XING-V-MURAD:value', 'IP2 (V)', 'orange', 'o', '-'),
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
                        markersize=4, linewidth=2, label=label, alpha=0.8, linestyle=linestyle)
                # Print final non-zero value
                final_val = data_nonzero.iloc[-1]
                print(f"  {label}: {final_val:.1f} μrad")

ax2.set_ylabel('Crossing Angle [μrad]', fontsize=18)
ax2.set_xlabel('Time (HH:MM)', fontsize=18)
ax2.set_title(f'{title_info}\nCrossing Angles at Main Interaction Points', fontsize=18, fontweight='bold')
ax2.legend(loc='best', fontsize=18, ncol=2)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
#axes fontsize
ax2.tick_params(axis='both', which='major', labelsize=18)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# %%
# PLOT 3: Beam Intensity and Emittance
print("\n" + "="*80)
print("PLOT 3: Beam Intensity and Emittance")
print("="*80)

fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Subplot 3a: Beam Intensity
if 'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY' in df.columns:
    data_b1 = df['LHC.BCTDC.A6R4.B1:BEAM_INTENSITY'].dropna()
    ax3a.plot(data_b1.index, data_b1.values, 'b-', linewidth=2, label='Beam 1', alpha=0.8)
    print(f"  Beam 1 intensity: {data_b1.iloc[0]:.2e} (start), {data_b1.iloc[-1]:.2e} (end)")

if 'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY' in df.columns:
    data_b2 = df['LHC.BCTDC.A6R4.B2:BEAM_INTENSITY'].dropna()
    ax3a.plot(data_b2.index, data_b2.values, 'r-', linewidth=2, label='Beam 2', alpha=0.8)
    print(f"  Beam 2 intensity: {data_b2.iloc[0]:.2e} (start), {data_b2.iloc[-1]:.2e} (end)")

ax3a.set_ylabel('Beam Intensity [protons]', fontsize=12)
ax3a.set_title(f'{title_info}\nBeam Intensity', fontsize=13, fontweight='bold')
ax3a.legend(loc='best', fontsize=11)
ax3a.grid(True, alpha=0.3)
ax3a.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Subplot 3b: Emittance
emittance_plotted = False
if 'EMITTANCE_SCAN_IP1:Emittance:HORIZONTAL_AverageEmittance_um' in df.columns:
    data_h = df['EMITTANCE_SCAN_IP1:Emittance:HORIZONTAL_AverageEmittance_um'].dropna()
    if len(data_h) > 0:
        ax3b.plot(data_h.index, data_h.values, 'b-', marker='o', markersize=5,
                 linewidth=2, label='Horizontal', alpha=0.8)
        emittance_plotted = True
        print(f"  Horizontal emittance: {data_h.mean():.2f} μm (mean)")

if 'EMITTANCE_SCAN_IP1:Emittance:VERTICAL_AverageEmittance_um' in df.columns:
    data_v = df['EMITTANCE_SCAN_IP1:Emittance:VERTICAL_AverageEmittance_um'].dropna()
    if len(data_v) > 0:
        ax3b.plot(data_v.index, data_v.values, 'r-', marker='s', markersize=5,
                 linewidth=2, label='Vertical', alpha=0.8)
        emittance_plotted = True
        print(f"  Vertical emittance: {data_v.mean():.2f} μm (mean)")

if emittance_plotted:
    ax3b.set_ylabel('Emittance [μm]', fontsize=12)
    ax3b.set_xlabel('Time (HH:MM)', fontsize=12)
    ax3b.set_title('Normalized Emittance (IP1)', fontsize=13, fontweight='bold')
    ax3b.legend(loc='best', fontsize=11)
    ax3b.grid(True, alpha=0.3)
else:
    ax3b.text(0.5, 0.5, 'No emittance data available',
             transform=ax3b.transAxes, ha='center', va='center',
             fontsize=14, color='gray')
    print("  No emittance data available")

ax3b.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax3b.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.setp(ax3b.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# %%
# PLOT 4: Luminosity at Main Experiments
print("\n" + "="*80)
print("PLOT 4: Total Instantaneous Luminosity")
print("="*80)

fig4, ax4 = plt.subplots(figsize=(14, 7))

experiments = ['ATLAS', 'CMS', 'LHCB', 'ALICE']
colors_lumi = ['blue', 'red', 'green', 'orange']
linestyles = ['-', '-', '--', '-.']

for exp, color, ls in zip(experiments, colors_lumi, linestyles):
    var_name = f'{exp}:LUMI_TOT_INST'
    if var_name in df.columns:
        data = df[var_name].dropna()
        if len(data) > 0:
            # Convert from Hz/ub to 10^34 cm^-2 s^-1
            lumi_converted = data.values * 1e-4
            ax4.plot(data.index, lumi_converted, color=color, linewidth=2.5,
                    linestyle=ls, label=exp, alpha=0.8)
            peak_lumi = lumi_converted.max()
            mean_lumi = lumi_converted.mean()
            print(f"  {exp}: Peak = {peak_lumi:.2f}, Mean = {mean_lumi:.2f} [×10³⁴ cm⁻² s⁻¹]")

ax4.set_ylabel('Luminosity [10³⁴ cm⁻² s⁻¹]', fontsize=12)
ax4.set_xlabel('Time (HH:MM)', fontsize=12)
ax4.set_title(f'{title_info}\nTotal Instantaneous Luminosity at Main Experiments', fontsize=13, fontweight='bold')
ax4.legend(loc='best', fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax4.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# %%
# PLOT 5: COMPREHENSIVE SUMMARY - All information combined
print("\n" + "="*80)
print("PLOT 5: Comprehensive Summary - All Information Combined")
print("="*80)

fig5 = plt.figure(figsize=(18, 12))
gs = fig5.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

# Overall title
fig5.suptitle(f'{title_info}\nComprehensive Fill Analysis', fontsize=16, fontweight='bold', y=0.995)

# Define colors and styles for summary plot (same as individual plots)
colors_lumi_summary = ['blue', 'red', 'green', 'orange']
linestyles_summary = ['-', '-', '--', '-.']

# Panel 1: Beta* at main IPs
ax_s1 = fig5.add_subplot(gs[0, 0])

# Add optics background
if 'LhcStateTracker:State:opticName' in df.columns and optics_colormap is not None:
    plot_time_windows_background(ax_s1, df['LhcStateTracker:State:opticName'].dropna(),
                                 colormap=optics_colormap, alpha=0.2)

for ip, color, marker in zip(ips, colors_beta, markers):
    var_name = f'HX:BETASTAR_{ip}'
    if var_name in df.columns:
        data = df[var_name].dropna()
        if len(data) > 0:
            ax_s1.plot(data.index, data.values, color=color, marker=marker,
                      markersize=3, linewidth=1.5, label=f'β* {ip}')
ax_s1.set_ylabel('β* [cm]', fontsize=10)
ax_s1.set_title('Beta* at Main IPs', fontsize=11, fontweight='bold')

# Only show beta* in legend, not optics
handles, labels = ax_s1.get_legend_handles_labels()
beta_handles = [h for h, l in zip(handles, labels) if 'β*' in l]
beta_labels = [l for l in labels if 'β*' in l]
ax_s1.legend(beta_handles, beta_labels, loc='best', fontsize=8)

ax_s1.grid(True, alpha=0.3)
ax_s1.tick_params(axis='x', labelsize=8)
ax_s1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Panel 2: Crossing Angles
ax_s2 = fig5.add_subplot(gs[0, 1])
for var_name, label, color, marker, linestyle in xing_vars:
    if var_name in df.columns:
        data = df[var_name].dropna()
        if len(data) > 0:
            # Mask zero values (angles not used)
            data_nonzero = data[data.abs() > 0.1]
            if len(data_nonzero) > 0:
                ax_s2.plot(data_nonzero.index, data_nonzero.values, color=color, marker=marker,
                          markersize=2, linewidth=1.5, label=label, linestyle=linestyle, alpha=0.8)
ax_s2.set_ylabel('Crossing Angle [μrad]', fontsize=10)
ax_s2.set_title('Crossing Angles', fontsize=11, fontweight='bold')
ax_s2.legend(loc='best', fontsize=7, ncol=2)
ax_s2.grid(True, alpha=0.3)
ax_s2.tick_params(axis='x', labelsize=8)
ax_s2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Panel 3: Beam Intensity
ax_s3 = fig5.add_subplot(gs[1, :])
if 'LHC.BCTDC.A6R4.B1:BEAM_INTENSITY' in df.columns:
    data_b1 = df['LHC.BCTDC.A6R4.B1:BEAM_INTENSITY'].dropna()
    ax_s3.plot(data_b1.index, data_b1.values, 'b-', linewidth=2, label='Beam 1')
if 'LHC.BCTDC.A6R4.B2:BEAM_INTENSITY' in df.columns:
    data_b2 = df['LHC.BCTDC.A6R4.B2:BEAM_INTENSITY'].dropna()
    ax_s3.plot(data_b2.index, data_b2.values, 'r-', linewidth=2, label='Beam 2')
ax_s3.set_ylabel('Beam Intensity [protons]', fontsize=10)
ax_s3.set_title('Beam Intensity', fontsize=11, fontweight='bold')
ax_s3.legend(loc='best', fontsize=9)
ax_s3.grid(True, alpha=0.3)
ax_s3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
ax_s3.tick_params(axis='x', labelsize=8)
ax_s3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Panel 4: Emittance
ax_s4 = fig5.add_subplot(gs[2, :])
if 'EMITTANCE_SCAN_IP1:Emittance:HORIZONTAL_AverageEmittance_um' in df.columns:
    data_h = df['EMITTANCE_SCAN_IP1:Emittance:HORIZONTAL_AverageEmittance_um'].dropna()
    if len(data_h) > 0:
        ax_s4.plot(data_h.index, data_h.values, 'b-', marker='o', markersize=4,
                  linewidth=2, label='Horizontal')
if 'EMITTANCE_SCAN_IP1:Emittance:VERTICAL_AverageEmittance_um' in df.columns:
    data_v = df['EMITTANCE_SCAN_IP1:Emittance:VERTICAL_AverageEmittance_um'].dropna()
    if len(data_v) > 0:
        ax_s4.plot(data_v.index, data_v.values, 'r-', marker='s', markersize=4,
                  linewidth=2, label='Vertical')
ax_s4.set_ylabel('Emittance [μm]', fontsize=10)
ax_s4.set_title('Normalized Emittance (IP1)', fontsize=11, fontweight='bold')
ax_s4.legend(loc='best', fontsize=9)
ax_s4.grid(True, alpha=0.3)
ax_s4.tick_params(axis='x', labelsize=8)
ax_s4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Panel 5: Luminosity
ax_s5 = fig5.add_subplot(gs[3, :])
for exp, color, ls in zip(experiments, colors_lumi_summary, linestyles_summary):
    var_name = f'{exp}:LUMI_TOT_INST'
    if var_name in df.columns:
        data = df[var_name].dropna()
        if len(data) > 0:
            lumi_converted = data.values * 1e-4
            ax_s5.plot(data.index, lumi_converted, color=color, linewidth=2,
                      linestyle=ls, label=exp)
ax_s5.set_ylabel('Luminosity [10³⁴ cm⁻² s⁻¹]', fontsize=10)
ax_s5.set_xlabel('Time (HH:MM)', fontsize=10)
ax_s5.set_title('Total Instantaneous Luminosity', fontsize=11, fontweight='bold')
ax_s5.legend(loc='best', fontsize=9)
ax_s5.grid(True, alpha=0.3)
ax_s5.tick_params(axis='x', labelsize=8)
ax_s5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax_s5.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.show()

# %%
# Print final summary
print("\n" + "="*80)
print(f"FILL {fill_number} - ANALYSIS COMPLETE")
print("="*80)
print(f"\nFill Information:")
print(f"  Fill start time: {fill_start}")
print(f"  Analysis window: {analysis_start} to {analysis_end}")
print(f"  Analysis duration: {analysis_duration_hours:.2f} hours")
print("\nDone!")

# %%
