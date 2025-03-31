import matplotlib.pyplot as plt
import pandas as pd
import os

# get absolute path of the project dir
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# get path performance tracking subfolder
tracking_path = os.path.join(project_folder ,"unified_experiment/performance_tracking")

# get file paths of model (alias) tracking
path_champion = os.path.join(tracking_path ,"performance_data_champion.csv")
path_challenger = os.path.join(tracking_path ,"performance_data_challenger.csv")
path_baseline = os.path.join(tracking_path ,"performance_data_baseline.csv")

# open files
df_champion = pd.read_csv(path_champion)
df_challenger = pd.read(path_challenger)
df_baseline = pd.read_csv(path_baseline)


print(df_champion)
