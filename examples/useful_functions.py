import matplotlib.pyplot as plt


def plot_time_windows_background(ax, df_column, colormap=None, alpha=0.3, zorder=-1):
    """
    Plot colored vertical bars as a background for time series plots.
    Each bar represents a time window where the value in df_column is constant.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on
    df_column : pd.Series
        A pandas Series with DatetimeIndex and values (scalars or strings)
    colormap : dict or None, optional
        Dictionary mapping values to colors. If None, uses a default colormap.
        Example: {'INJPHYS': 'lightblue', 'STABLE': 'lightgreen'}
    alpha : float, optional
        Transparency of the bars (0-1), default 0.3
    zorder : int, optional
        Z-order for the bars (negative values place them behind other plot elements)

    Returns
    -------
    dict
        Dictionary mapping values to colors used
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

    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    return colormap