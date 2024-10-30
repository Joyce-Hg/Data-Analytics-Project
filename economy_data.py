import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv('country_data.csv')

# Step 2: Set the 'Year' column as the index for easier plotting
data.set_index('Year', inplace=True)
plt.rcParams.update({'font.size': 16}) 
# Step 3: Plot each variable as a separate line plot
variables = data.columns  # This will get all column names excluding 'Year'
for variable in variables:
     plt.figure(figsize=(10, 6))
     plt.plot(data.index, data[variable], marker='o', linestyle='-')
     plt.title(f'Trend of {variable} Over Time',fontsize=20, pad=20)
     plt.xlabel('Year')
     plt.ylabel(variable)
     plt.grid(True)
     plt.show()
