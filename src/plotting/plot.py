import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
random_file = 'data/STOCHASTIC_csv_conn0_ep2.csv'
dqn_file = 'data/DQN_2way_conn0_ep2.csv'
a2c_file = 'data/A2C_2way_conn0_ep2.csv'

# Read the csv files
dqn_data = pd.read_csv(dqn_file)
a2c_data = pd.read_csv(a2c_file)
random_data = pd.read_csv(random_file)

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

