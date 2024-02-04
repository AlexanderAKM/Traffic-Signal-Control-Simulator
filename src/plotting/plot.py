import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def average_csv(files):
    """Averages multiple CSV files into a single DataFrame."""
    data_frames = [pd.read_csv(f) for f in files]
    # Ensure all CSV files have the same number of rows (steps)
    min_length = min(len(df) for df in data_frames)
    trimmed_data = [df.iloc[:min_length] for df in data_frames]
    concatenated = pd.concat(trimmed_data)
    mean_data = concatenated.groupby(concatenated.index).mean()
    return mean_data

def smooth_and_ci(data, window_size=10):
    """Applies smoothing and calculates the confidence interval."""
    smoothed = data.rolling(window=window_size, min_periods=1, center=True).mean()
    std = data.rolling(window=window_size, min_periods=1, center=True).std(ddof=0)
    ci = 1.96 * std / np.sqrt(window_size)  # 95% confidence interval
    return smoothed, ci

# File paths
dqn_files = ['data/DQN_2way_ep2.csv', 'data/DQN_2way_ep3.csv', 'data/DQN_2way_ep4.csv', 'data/DQN_2way_ep5.csv']
a2c_files = ['data/A2C_2way_ep1.csv', 'data/A2C_2way_ep2.csv', 'data/A2C_2way_ep3.csv', 'data/A2C_2way_ep4.csv', 'data/A2C_2way_ep5.csv']
random_files = ['data/RANDOM_2way_0_ep1.csv', 'data/RANDOM_2way_1_ep1.csv', 'data/RANDOM_2way_2_ep1.csv', 'data/RANDOM_2way_3_ep1.csv', 'data/RANDOM_2way_4_ep1.csv']

# Averaging
dqn_avg = average_csv(dqn_files)
a2c_avg = average_csv(a2c_files)
random_avg = average_csv(random_files)

# Smoothing and CI
dqn_smoothed, dqn_ci = smooth_and_ci(dqn_avg['system_total_waiting_time'])
a2c_smoothed, a2c_ci = smooth_and_ci(a2c_avg['system_total_waiting_time'])
random_smoothed, random_ci = smooth_and_ci(random_avg['system_total_waiting_time'])

# Adjust steps based on the actual range in the CSV files
steps = dqn_avg['step']

# Plotting DQN vs A2C.
plt.figure(figsize=(12, 6))
plt.fill_between(steps, dqn_smoothed - dqn_ci, dqn_smoothed + dqn_ci, color='blue', alpha=0.2)
plt.plot(steps, dqn_smoothed, label='DQN', color='blue')
plt.fill_between(steps, a2c_smoothed - a2c_ci, a2c_smoothed + a2c_ci, color='orange', alpha=0.2)
plt.plot(steps, a2c_smoothed, label='A2C', color='orange')
plt.xlabel('Step')
plt.ylabel('System Total Waiting Time')
plt.title('Smoothed System Total Waiting Time vs Step (with CI)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data/DQNvsA2C_waiting_time_with_ci.png', format='png')
plt.show()
plt.close()

# Plotting DQN vs Stochastic.
plt.figure(figsize=(12, 6))
plt.fill_between(steps, dqn_smoothed - dqn_ci, dqn_smoothed + dqn_ci, color='blue', alpha=0.2)
plt.plot(steps, dqn_smoothed, label='DQN', color='blue')
plt.fill_between(steps, random_smoothed - random_ci, random_smoothed + random_ci, color='orange', alpha=0.2)
plt.plot(steps, random_smoothed, label='RANDOM', color='orange')
plt.xlabel('Step')
plt.ylabel('System Total Waiting Time')
plt.title('Smoothed System Total Waiting Time vs Step (with CI)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data/DQNvsRANDOM_waiting_time_with_ci.png', format='png')
plt.show()
plt.close()

# Plotting RANDOM vs A2C.
plt.figure(figsize=(12, 6))
plt.fill_between(steps, random_smoothed - random_ci, random_smoothed + random_ci, color='blue', alpha=0.2)
plt.plot(steps, random_smoothed, label='RANDOM', color='blue')
plt.fill_between(steps, a2c_smoothed - a2c_ci, a2c_smoothed + a2c_ci, color='orange', alpha=0.2)
plt.plot(steps, a2c_smoothed, label='A2C', color='orange')
plt.xlabel('Step')
plt.ylabel('System Total Waiting Time')
plt.title('Smoothed System Total Waiting Time vs Step (with CI)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data/RANDOMvsA2C_waiting_time_with_ci.png', format='png')
plt.show()
plt.close()


'''

def plotWaitingTime(random_file = None, dqn_file = None, a2c_file = None):
    
    # Read the csv files
    random_data = pd.read_csv(random_file)
    dqn_data = pd.read_csv(dqn_file)
    a2c_data = pd.read_csv(a2c_file)

    # Plot 1: DQN vs A2C
    plt.figure(figsize=(10, 5))
    plt.plot(dqn_data['step'], dqn_data['system_total_waiting_time'], label='DQN')
    plt.plot(a2c_data['step'], a2c_data['system_total_waiting_time'], label='A2C')

    plt.xlabel('Step')
    plt.ylabel('System Total Waiting Time')
    plt.title('System Total Waiting Time vs Step')
    plt.legend()
    plt.grid(True)

    plt.savefig('data/DQNvsA2C_waiting_time.png', format='png')

    plt.close()

    # Plot 2: DQN
    plt.figure(figsize=(10, 5))
    plt.plot(dqn_data['step'], dqn_data['system_total_waiting_time'], label='DQN')

    plt.xlabel('Step')
    plt.ylabel('System Total Waiting Time')
    plt.title('System Total Waiting Time vs Step')
    plt.legend()
    plt.grid(True)

    plt.savefig('data/DQN_waiting_time.png', format='png')

    plt.close()

    # Plot 3: A2C
    plt.figure(figsize=(10, 5))
    plt.plot(a2c_data['step'], a2c_data['system_total_waiting_time'], label='A2C')

    plt.xlabel('Step')
    plt.ylabel('System Total Waiting Time')
    plt.title('System Total Waiting Time vs Step')
    plt.legend()
    plt.grid(True)

    plt.savefig('data/A2C_waiting_time.png', format='png')

    plt.close()

    # Plot 4: DQN vs Random
    plt.figure(figsize=(10, 5))
    plt.plot(dqn_data['step'], dqn_data['system_total_waiting_time'], label='DQN')
    plt.plot(random_data['step'], random_data['system_total_waiting_time'], label='Random')

    plt.xlabel('Step')
    plt.ylabel('System Total Waiting Time')
    plt.title('System Total Waiting Time vs Step')
    plt.legend()
    plt.grid(True)

    plt.savefig('data/DQNvsRandom_waiting_time.png', format='png')

    plt.close()

    # Plot 5: A2C vs Random
    plt.figure(figsize=(10, 5))
    plt.plot(a2c_data['step'], a2c_data['system_total_waiting_time'], label='A2C')
    plt.plot(random_data['step'], random_data['system_total_waiting_time'], label='Random')

    plt.xlabel('Step')
    plt.ylabel('System Total Waiting Time')
    plt.title('System Total Waiting Time vs Step')
    plt.legend()
    plt.grid(True)

    plt.savefig('data/A2CvsRandom_waiting_time.png', format='png')

    plt.close()
'''
