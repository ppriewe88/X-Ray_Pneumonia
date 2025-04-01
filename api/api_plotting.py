import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
import time


def generate_model_comparison_plot(target = "accuracy_last_50_predictions", scaling =  "log_counter"):

    # get absolute path of the project dir
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # get path performance tracking subfolder
    tracking_path = os.path.join(project_folder ,"unified_experiment/performance_tracking")

    # get file paths of model (alias) tracking
    path_champion = os.path.join(tracking_path ,"performance_data_champion.csv")
    path_challenger = os.path.join(tracking_path ,"performance_data_challenger.csv")
    path_baseline = os.path.join(tracking_path ,"performance_data_baseline.csv")

    # open files as dataframes
    df_champion = pd.read_csv(path_champion)
    df_challenger = pd.read_csv(path_challenger)
    df_baseline = pd.read_csv(path_baseline)

    # convert timestamp to datetime
    df_champion['timestamp'] = pd.to_datetime(df_champion['timestamp'])
    df_challenger['timestamp'] = pd.to_datetime(df_challenger['timestamp'])
    df_baseline['timestamp'] = pd.to_datetime(df_baseline['timestamp'])

    # create fig and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    x_axis = scaling
    # plot thre dataframes as traces
    ax.plot(df_champion[x_axis], df_champion[target], label='Champion', color='green', linestyle='-', linewidth=2)
    # ax.plot(df_challenger[x_axis], df_challenger[target], label='Challenger', color='blue', linestyle='--', linewidth=2)
    ax.plot(df_baseline[x_axis], df_baseline[target], label='Baseline', color='red', linestyle=':', linewidth=2)

    # set common axis labels and titles
    ax.set_ylabel(target, fontsize=12)
    ax.set_title(f'Model comparison over time ({target})', fontsize=14)

    # legend
    ax.legend(fontsize=10)

    # set custom axis and title formatting according to scaling
    if scaling == "timestamp":
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        ax.set_xlabel("Time of run", fontsize=12)

    else:
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_major_locator(plt.AutoLocator())
        ax.set_xlabel("Run number", fontsize=12)

    # add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # show plot
    plt.tight_layout()

    return fig


# keys: ['log_counter', 'timestamp', 'y_true', 'y_pred', 'accuracy',
#           'cumulative_accuracy', 'global_accuracy',
#          'accuracy_last_25_predictions', 'filename', 'model_version',
#           'model_tag', 'model_alias']

# scaling: timestamp, log_counter
# start = time.time()
# generate_model_comparison_plot(target="global_accuracy", scaling = "log_counter")
# end = time.time()
# print("plot generation runtime: ", end-start)
# generate_model_comparison_plot(target="accuracy_last_50_predictions", scaling = "log_counter")
# generate_model_comparison_plot(target="global_accuracy", scaling = "timestamp")
# generate_model_comparison_plot(target="accuracy_last_25_predictions", scaling = "timestamp")