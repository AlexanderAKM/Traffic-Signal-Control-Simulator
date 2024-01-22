import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv('data/test_csv_conn0_ep1.csv')

# Assuming the CSV file has columns named 'time' and 'system_total_waiting_time'
# And you want to plot the data up to time 3700


# Plotting
plt.figure(figsize=(10, 5))  # Change the figure size as needed
plt.plot(df['step'], df['system_total_waiting_time'], label='Total Waiting Time')

# Labeling the axes
plt.xlabel('Step')
plt.ylabel('System Total Waiting Time')

# Title for the plot
plt.title('System Total Waiting Time vs. Time')

# Display legend
plt.legend()

# Show grid
plt.grid(True)

# Save the plot as a file (optional)
plt.savefig('data/system_total_waiting_time_plot.png', format='png')

# Show the plot
plt.show()
