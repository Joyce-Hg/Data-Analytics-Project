import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Read CSV file
data = pd.read_csv('country_data.csv')

# Step 2: Control the period that we want to obserce
data_filtered = data[data['Year'] >= 2010]

# Step 3: Choose the specific variables we want to observe 
attributes = ['Job vacancy', 'Age>65', 'Labor participant rate', 'Labor productivity', 'Population']
X = data_filtered[['Age>65', 'Labor participant rate', 'Labor productivity', 'Population']]
y = data_filtered['Job vacancy']

# Step 4: Before doing Lasso Regression we need to standardize all the data. 
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
y_standardized = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Step 5: Cross-validation to choose most suitable Lasso Alpha 
lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5)
lasso_cv.fit(X_standardized, y_standardized)

# Step 6: Collect results for each alpha
results = []
for i, alpha in enumerate(lasso_cv.alphas_):
    # Fit a Lasso model with a specific alpha to get the coefficients for each alpha
    lasso_temp = LassoCV(alphas=[alpha], cv=5)
    lasso_temp.fit(X_standardized, y_standardized)
    row = {'alpha': alpha}
    for j, col_name in enumerate(X.columns):
        row[col_name] = lasso_temp.coef_[j]
    results.append(row)

# Step 7: Convert results to DataFrame and export to CSV
results_df = pd.DataFrame(results)


print("All Lasso results")
print(results_df)

# Step 8: Plot bar chart for alpha=0.1 to show variable importance
# Filter the results for alpha=0.1
coefficients = results_df[results_df['alpha'] == 0.1].iloc[0, 1:]

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 6))
bars=plt.bar(coefficients.index, coefficients.values, color='orange')
plt.xlabel("Variables", fontsize=14)
plt.ylabel("Coefficient Value", fontsize=14)
plt.title("Variable Importance at Alpha = 0.1",fontsize=18, pad=20)
plt.grid(axis='y')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), ha='center', va='bottom', fontsize=10)


plt.show()