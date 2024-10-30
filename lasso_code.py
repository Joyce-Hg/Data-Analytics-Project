import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

#in order to know more collinearity between different variables, we using VIF to do research

# Step 1: read CSV file
data = pd.read_csv('country_data.csv')

# Step 2: choose the duration of observation
# because we want to focus on the period in recent years , so we choode the data after 2010
data_filtered = data[data['Year'] >= 2010]

# Step 3: standardize all the variable data before using VIF
attributes = ['Age>65', 'Labor participant rate', 'Labor productivity','Population']
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(data_filtered[attributes]), columns=attributes)

# Step 4:  Calculating VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = data_standardized.columns
vif_data["VIF"] = [variance_inflation_factor(data_standardized.values, i) for i in range(data_standardized.shape[1])]

# Step 5: Print and save the result
print("VIF for each variable:")
print(vif_data)
vif_data.to_csv('VIF_results_cleaned.csv', index=False)


# Step 6: Plot VIF as a bar chart
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 6))
bars = plt.bar(vif_data["Variable"], vif_data["VIF"], color=['orange' if vif > 5 else 'blue' for vif in vif_data["VIF"]])
plt.axhline(y=5, color='gray', linestyle='--', label='Threshold (VIF = 5)')
plt.xlabel("Variables",fontsize=14)
plt.ylabel("VIF Value",fontsize=14)
plt.title("Variance Inflation Factor (VIF) for Each Variable",fontsize=18, pad=20)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()