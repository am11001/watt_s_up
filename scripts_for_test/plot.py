from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


def plot_distrib(df_real, df_synth):
    fig = plt.figure(figsize = (7, 5))
    plt.hist(df_real.to_numpy().flatten(), bins = 100, alpha = 0.5, label = 'Real', color = 'aqua')
    plt.hist(df_synth.to_numpy().flatten(), bins = 100, alpha = 0.5, label = 'Synthetic', color = 'hotpink')
    plt.title('Value distributions', fontweight = 'bold')
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.close()
    return fig

########################################################################################################################

def plot_stat(stat, df_real, df_synth, ax, title, descrFontSize = 7):
    box_dict = ax.boxplot([getattr(df_real, stat)(), getattr(df_synth, stat)()], vert = True)
    ax.set_xticklabels(['real', 'synthetic'])
    ax.set_title(title, fontweight = 'bold')
    ax.set_ylabel('value')
    ax.grid()
    for idx, box in enumerate(box_dict['boxes']):
        x_pos = idx + 1
        q1 = box.get_path().vertices[0, 1]
        q3 = box.get_path().vertices[2, 1]
        whiskers = [line.get_ydata()[1] for line in box_dict['whiskers'][idx*2:idx*2 + 2]]
        medians = box_dict['medians'][idx].get_ydata()[0]
        ax.text(x_pos + 0.1, q1, f'Q1: {q1:.2f}', va = 'center', fontsize = descrFontSize, color = 'blue')
        ax.text(x_pos + 0.1, q3, f'Q3: {q3:.2f}', va = 'center', fontsize = descrFontSize, color = 'blue')
        ax.text(x_pos + 0.1, medians, f'Med: {medians:.2f}', va='center', fontsize = descrFontSize, color='red')
        ax.text(x_pos + 0.1, whiskers[0], f'Min: {whiskers[0]:.2f}', va = 'center', fontsize = descrFontSize, color = 'green')
        ax.text(x_pos + 0.1, whiskers[1], f'Max: {whiskers[1]:.2f}', va = 'center', fontsize = descrFontSize, color = 'green')


def plot_mean(df_real, df_synth, ax):
    plot_stat('mean', df_real, df_synth, ax, 'mean values')


def plot_std(df_real, df_synth, ax):
    plot_stat('std', df_real, df_synth, ax, 'standard deviation values')


def plot_median(df_real, df_synth, ax):
    plot_stat('median', df_real, df_synth, ax, 'median values')


def plot_min(df_real, df_synth, ax):
    plot_stat('min', df_real, df_synth, ax, 'minimum values')


def plot_max(df_real, df_synth, ax):
    plot_stat('max', df_real, df_synth, ax, 'maximum values')


def plot_skew(df_real, df_synth, ax):
    plot_stat('skew', df_real, df_synth, ax, 'skew values')


def plot_stats(df_real, df_synth):
    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (20, 10))
    axes = axes.flatten()
    plotFuncs = [plot_mean, plot_std, plot_median, plot_min, plot_max, plot_skew]
    for func, ax in zip(plotFuncs, axes):
        func(df_real, df_synth, ax)
    plt.suptitle('Comparison of...', ha = 'center', fontsize = 16, fontweight = 'bold')
    plt.tight_layout()
    plt.close()
    return fig

########################################################################################################################

def plot_mean_profiles(df_real, df_synth):
    arr_real = df_real.to_numpy()
    arr_synth = df_synth.to_numpy()
    maxCols = min([arr_real.shape[1], arr_synth.shape[1]])
    fig, axs = plt.subplots(ncols = 2, nrows = 2, figsize = (12, 8))
    sns.heatmap(arr_real.mean(axis = 1).reshape(24, -1), ax = axs[0, 0])
    sns.heatmap(arr_synth.mean(axis = 1).reshape(24, -1), ax = axs[0, 1])
    sns.heatmap((arr_synth[:, :maxCols] - arr_real[:, :maxCols]).mean(axis = 1).reshape(24, -1), ax = axs[1, 0])
    axs[0, 0].set_title('Mean real profile', fontweight = 'bold')
    axs[0, 1].set_title('Mean synthetic profile', fontweight = 'bold')
    axs[1, 0].set_title('Mean difference profile (synthetic - real)', fontweight = 'bold')
    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.close()
    return fig

########################################################################################################################

def create_df_trend(df_dict, res, stats = ['mean', 'std', 'median', 'min', 'max', 'skew']):
    dfs = []
    for type, df in df_dict.items():
        if res == 'week':
            df.index.week = df.index.isocalendar().week
        df_group = df.groupby([getattr(df.index, res)]).agg(stats)
        df_group = df_group.T.groupby(level = 1).mean()
        df_group.columns = np.arange(df_group.shape[1])
        df_group = df_group.T
        df_group['type'] = type
        df_group['time'] = df_group.index
        dfs.append(df_group)
    df_trend = pd.concat(dfs, axis = 0)
    return df_trend


def plot_mean_trends(df_trend, res, stats = ['mean', 'std', 'median', 'min', 'max', 'skew']):
    df_trend = df_trend.melt(id_vars = ['type', 'time'], value_vars = stats, var_name = 'statistic', value_name = 'value')
    df_trend['time'] = df_trend['time'].astype(int, errors = 'raise')
    fig = sns.FacetGrid(df_trend, col = 'statistic', col_wrap = 3, sharex = False, sharey = False, height = 3.26, aspect = 1.5)
    fig.map_dataframe(sns.lineplot, x = 'time', y = 'value', hue = 'type', style = 'type')
    fig.set_axis_labels('time')
    fig.axes.flat[0].set_ylabel('value')
    fig.axes.flat[3].set_ylabel('value')
    fig.axes.flat[0].legend()
    plt.suptitle(f'{res.capitalize()}ly trend'.replace('Day', 'Dai'), fontweight = 'bold')
    plt.tight_layout()
    plt.close()
    return fig.fig


def plot_RMSE(df_trend, res):
    df_real = df_trend.loc[df_trend['type'] == 'Real', :].drop(columns = ['type', 'time'])
    df_synth = df_trend.loc[df_trend['type'] == 'Synthetic', :].drop(columns = ['type', 'time'])
    rmse = ((df_real - df_synth)**2).mean().apply(lambda x: np.sqrt(x))
    df_rmse = (pd.DataFrame(rmse, columns = ['value']).reset_index().rename(columns = {'index': 'statistic'}))
    fig, ax = plt.subplots()
    sns.barplot(data = df_rmse, x = 'statistic', y = 'value', ax = ax)
    for container in ax.containers:
        ax.bar_label(container, fmt = '%.3f', fontsize = 7, padding = 3)
    ax.set_title(f'RMSE ({res}ly trend)'.replace('day', 'dai'), fontweight = 'bold')
    ax.set_ylabel('value')
    plt.tight_layout()
    plt.close()
    return df_rmse, fig

########################################################################################################################

def create_plots(df_real, df_synth, outputPath = Path.cwd() / 'plots'):
    fig_dict = {}
    rmse_dict = {}

    # Value distributions
    fig_dict['distrib_all_profiles'] = plot_distrib(df_real, df_synth)

    # Various statistics
    fig_dict['stats_all_profiles'] = plot_stats(df_real, df_synth)

    # Mean profiles
    fig_dict['mean_profiles'] = plot_mean_profiles(df_real, df_synth)

    df_dict = {'Real': df_real, 'Synthetic': df_synth}
    for res in ['hour', 'day', 'week', 'month']:
        df_trend = create_df_trend(df_dict, res)

        # Mean trends
        fig_dict[f'{res}ly_trend'.replace('day', 'dai')] = plot_mean_trends(df_trend, res)

        # RMSE
        rmse_dict[f'RMSE_{res}ly_trend'.replace('day', 'dai')], fig_dict[f'RMSE_{res}ly_trend'.replace('day', 'dai')] = plot_RMSE(df_trend, res)

    outputPath = outputPath / datetime.today().strftime('%Y-%m-%d_%H%M')
    outputPath.mkdir(parents = True, exist_ok = True)
    for key, value in fig_dict.items():
        value.savefig(outputPath / f'{key}.png', bbox_inches = 'tight')
    return fig_dict, rmse_dict