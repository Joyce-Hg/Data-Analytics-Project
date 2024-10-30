import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler

# Analyze vacancy data by each industry.
# Data download and preprocessing.

industry_data=pd.read_csv("industry_data.csv",sep=",")
print(industry_data.head(5))
print(industry_data.info())
industry_data["Total Vacancy"]=pd.to_numeric(industry_data["Total Vacancy"], errors="coerce")
industry_data=industry_data.dropna(subset=["Total Vacancy"])
industry_data["Total Vacancy"].astype(float)
industry_data_sorted=industry_data.sort_values(by="Year", ascending=True)
print(industry_data_sorted.head(5))


# Using linear regression models to assess the impact of industry vacancy on total job vacancies. 
# X: YoY change in the vacancy number for a specific industry
# y: Year-over-year (YoY) change in the total vacancy number

industry_data_country=industry_data_sorted.groupby("Year")["Total Vacancy"].sum().reset_index(name="Total_Vacancies")
industry_data_country["Total_Percent_Increase"]=industry_data_country["Total_Vacancies"].pct_change().fillna(0)
industry_data_industry=industry_data_sorted.groupby(["Industry", "Year"])["Total Vacancy"].sum().reset_index()
industry_data_industry["Vacancy_Percent_Increase"]=industry_data_industry.groupby("Industry")["Total Vacancy"].pct_change().fillna(0)
industry_data_industry_merged=industry_data_industry.pivot(index="Year", columns="Industry", values="Vacancy_Percent_Increase").reset_index()
industry_data_industry_merged=pd.merge(industry_data_industry_merged, industry_data_country[["Year","Total_Percent_Increase"]], on="Year")

X=industry_data_industry_merged.drop(columns=["Year","Total_Percent_Increase"])
y=industry_data_industry_merged["Total_Percent_Increase"]

# Standardize X and y
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Run linear regression
model=LinearRegression()
model.fit(X_scaled,y_scaled)
coefficients=pd.DataFrame({"Industry": X_scaled.columns, "Coefficient": model.coef_})
print("coefficients: ")
print(coefficients)
print("\nR^2: ", model.score(X_scaled,y_scaled))

y = y_scaled  
results_df = pd.DataFrame(columns=["Industry", "Coefficient", "Intercept", "R_squared"])
model = LinearRegression()

# Loop through each industry in the standardized X data
for industry in X_scaled.columns:
    X = X_scaled[[industry]]  
    
    model.fit(X, y)
    coefficient = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)

    temp_df = pd.DataFrame({
        "Industry": [industry],
        "Coefficient": [coefficient],
        "Intercept": [intercept],
        "R_squared": [r_squared]
    })
    results_df = pd.concat([results_df, temp_df], ignore_index=True)

print(results_df)

# Data visualization
# Plot R_squared chart
results_df_sorted = results_df.sort_values(by="R_squared", ascending=False)
colors = ["skyblue" if r2 <= 0.8 else "salmon" for r2 in results_df_sorted["R_squared"]]
plt.figure(figsize=(14, 10)) 
plt.rc("axes", titlesize=18)  
plt.rc("axes", labelsize=16)  
plt.rc("xtick", labelsize=14) 
plt.rc("ytick", labelsize=12) 
bars = plt.barh(results_df_sorted["Industry"], results_df_sorted["R_squared"], color=colors)
plt.xlabel("R Squared")
plt.ylabel("Industry")
plt.title("R Squared Values by Industry (Highlighting R^2 > 0.8)")
plt.gca().invert_yaxis()  
plt.axvline(x=0.8, color="gray", linestyle="--", linewidth=1)

for index, value in enumerate(results_df_sorted["R_squared"]):
    plt.text(value + 0.01, index, f"{value:.2f}", va="center", fontsize=12) 
plt.tight_layout()
plt.show()

# Plot Coefficient Chart

results_df_sorted = results_df.sort_values(by="Coefficient", ascending=False)
colors = ["salmon" if coef > 0.9 else "skyblue" for coef in results_df_sorted["Coefficient"]]
plt.figure(figsize=(14, 10))  
plt.rc("axes", titlesize=18)  
plt.rc("axes", labelsize=16)  
plt.rc("xtick", labelsize=14)  
plt.rc("ytick", labelsize=14)  
bars = plt.barh(results_df_sorted["Industry"], results_df_sorted["Coefficient"], color=colors)
plt.xlabel("Coefficient")
plt.ylabel("Industry")
plt.title("Regression Coefficients by Industry (Highlighting Coefficients > 0.9)")

for index, value in enumerate(results_df_sorted["Coefficient"]):
    plt.text(value, index, f"{value:.2f}", va="center", fontsize=12)  
plt.gca().invert_yaxis()
plt.axvline(x=0.9, color="gray", linestyle="--", linewidth=1)
plt.tight_layout()
plt.show()

# Plot the vacancy trend for each industry with high r squared

selected_industry=["Other economic services", "Traffic and storage", "Manufacturing", 
"Other services", "Trade", "Scientific & Tech. services"]

plt.rc("axes", titlesize=20)  
plt.rc("axes", labelsize=18)  
plt.rc("xtick", labelsize=16) 
plt.rc("ytick", labelsize=16) 
plt.rc("legend", fontsize=12) 
palette = sns.color_palette("husl", len(selected_industry))
plt.figure(figsize=(14, 10))  

for idx, industry in enumerate(selected_industry):
    industry_data = industry_data_industry[industry_data_industry["Industry"] == industry]
    
    if industry == "Scientific & Tech. services":
        plt.plot(
            industry_data["Year"], industry_data["Total Vacancy"],
            label=industry, color="red", linestyle="-", linewidth=2.5, marker="o"
        )
    else:
        plt.plot(
            industry_data["Year"], industry_data["Total Vacancy"],
            label=industry, color=palette[idx], linestyle="--", linewidth=1.5, marker="o"
        )

plt.title("Yearly Vacancy Numbers for Industries with High R^2 Values (2010-2023)")
plt.xlabel("Year")
plt.ylabel("Total Vacancy")
plt.xticks(rotation=45)  
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=0., frameon=False)
plt.grid(True) 
plt.show()
