import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import Lasso
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import seaborn as sns


# Step 1: read CSV file
data = pd.read_csv('country_data.csv')

# step 2: understanding the correlation of each variables
correlation_matrix = data.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Step 3: standardized all the variables  
attributes = ['Job vacancy', 'Net migration','Age>65', 'Labor participant rate', 'Labor productivity']
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(data[attributes]), columns=attributes)

correlation_matrix_standardized = data_standardized.corr()
print("Correlation Matrix after standardized:")
print(correlation_matrix_standardized)



# Step 4: draw the scatter plots and heat map
scatter_matrix(data_standardized, figsize=(36, 30), diagonal='hist')
plt.show()


attributes = ['Job vacancy', 'Investment ratio','Population','Birth rate']
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(data[attributes]), columns=attributes)

correlation_matrix_standardized = data_standardized.corr()
print("Correlation Matrix after standardized:")
print(correlation_matrix_standardized)
scatter_matrix(data_standardized, figsize=(36, 30), diagonal='hist')
plt.show()



attributes=['Job vacancy','Net migration','Age>65', 'Labor participant rate', 'Labor productivity', 'Investment ratio','Population','Birth rate']
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(data[attributes]), columns=attributes)
correlation_matrix_standardized = data_standardized.corr()
plt.figure(figsize=(16, 6))
sns.heatmap(correlation_matrix_standardized, annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.show()



# Step 5: observe recent data to see the trend sustain or not
# we only want to obserce the years after 2010
recent_data = data[data['Year'] >= 2010]

attributes = ["Job vacancy", "Age>65", "Labor participant rate", "Labor productivity","Population"]
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(recent_data[attributes]), columns=attributes)

correlation_matrix_standardized = data_standardized.corr()
print("Correlation Matrix after standardized:")
print(correlation_matrix_standardized)
scatter_matrix(data_standardized, figsize=(36, 30), diagonal='hist')
plt.show()

plt.figure(figsize=(16, 6))
sns.heatmap(correlation_matrix_standardized, annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.show()




