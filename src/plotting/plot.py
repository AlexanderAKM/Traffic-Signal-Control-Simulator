import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
stochastic_file = 'data/2way_stochastic_csv_run0_conn0_ep1.csv'
ppo_file = 'data/PPO_2way_test_csv_run0_conn0_ep1.csv'
dqn_file = 'data/dqn_2way_test_csv_run0_conn0_ep2.csv'
a2c_file = 'data/A2C_2way_test_csv_run0_conn0_ep2.csv'

# Read the csv files
dqn_data = pd.read_csv(dqn_file)
a2c_data = pd.read_csv(a2c_file)
ppo_data = pd.read_csv(ppo_file)
stochastic_data = pd.read_csv(stochastic_file)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(dqn_data['step'], dqn_data['system_total_waiting_time'], label='DQN')
plt.plot(a2c_data['step'], a2c_data['system_total_waiting_time'], label='A2C')
plt.plot(ppo_data['step'], ppo_data['system_total_waiting_time'], label='PPO')
plt.plot(stochastic_data['step'], stochastic_data['system_total_waiting_time'], label='Random')

# Label the axis 
plt.xlabel('Step')
plt.ylabel('System Total Waiting Time')
plt.title('System Total Waiting Time vs Step')
plt.legend()
plt.grid(True)

plt.savefig('outputs/total_waiting_time.png', format='png')

plt.show()