import pandas as pd
import matplotlib.pyplot as plt
import glob

# Define a function to calculate the rolling average
def rolling_average(data, window_size=2):
    return data.rolling(window=window_size, min_periods=1).mean()

# File pattern for the CSV files
file_pattern = 'data/test_csv_run*.csv'
files = glob.glob(file_pattern)

# Initialize a DataFrame to hold all the data
summed_data = None

# Loop through all files and sum the waiting times
for file in files:
    df = pd.read_csv(file)
    if summed_data is None:
        summed_data = df['system_total_waiting_time']
    else:
        summed_data += df['system_total_waiting_time']

# Calculate the rolling average for the summed data
summed_data_ra = rolling_average(summed_data, window_size=2)  # Adjust window size as needed

# Plotting
plt.figure(figsize=(14, 7))  # Adjust figure size to your preference
plt.plot(df['step'], summed_data_ra, label='Total Waiting Time (Smoothed)')

# Labeling the axes
plt.xlabel('Step')
plt.ylabel('Summed System Total Waiting Time (Smoothed)')

# Title for the plot
plt.title('Smoothed System Total Waiting Time vs. Time')

# Display legend
plt.legend()

# Show grid
plt.grid(True)

# Save the plot as a file (optional)
plt.savefig('data/smoothed_total_waiting_time_plot.png', format='png')

# Show the plot
plt.show()
