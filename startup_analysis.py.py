# ==========================================
# INDIAN STARTUP FUNDING ANALYSIS (FINAL)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------
file_path = r"D:\indian-startup-funding-analysis\archive(4)\startup_funding.csv"
df = pd.read_csv(file_path, encoding="latin-1")

print("Data Loaded Successfully")
print(df.head())

# ------------------------------------------------
# 2. CLEAN COLUMN NAMES
# ------------------------------------------------
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

df.rename(columns={
    "ï»¿sr_no": "sr_no",
    "date_dd/mm/yyyy": "date",
    "startup_name": "startup",
    "industry_vertical": "industry",
    "investors_name": "investors",
    "investmentntype": "investment_type",
    "amount_in_usd": "amount"
}, inplace=True)

# ------------------------------------------------
# 3. BASIC CLEANING
# ------------------------------------------------
df.drop_duplicates(inplace=True)

df = df[df['amount'].notna()]
df = df[df['amount'] != "Undisclosed"]

df['amount'] = df['amount'].astype(str)
df['amount'] = df['amount'].str.replace(',', '')
df['amount'] = df['amount'].str.replace('+', '')
df['amount'] = df['amount'].str.replace(' ', '')
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

df = df.dropna(subset=['amount'])

# ------------------------------------------------
# 4. DATE PROCESSING
# ------------------------------------------------
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

# ------------------------------------------------
# 5. CLEAN TEXT COLUMNS
# ------------------------------------------------
df['industry'] = df['industry'].fillna("Unknown")
df['investment_type'] = df['investment_type'].fillna("Unknown")
df['investors'] = df['investors'].fillna("Unknown")

print("\nCleaned Data Shape:", df.shape)

# ------------------------------------------------
# 6. FUNDING TREND BY YEAR
# ------------------------------------------------
funding_year = df.groupby('year')['amount'].sum().sort_index()

plt.figure()
plt.plot(funding_year.index, funding_year.values, marker='o')
plt.title("Total Startup Funding by Year")
plt.xlabel("Year")
plt.ylabel("Funding (USD)")
plt.show()

# ------------------------------------------------
# 7. TOP INDUSTRIES BY FUNDING
# ------------------------------------------------
top_industries = df.groupby('industry')['amount'].sum().sort_values(ascending=False).head(10)

plt.figure()
top_industries.plot(kind='bar')
plt.title("Top Industries by Funding")
plt.ylabel("Funding (USD)")
plt.show()

# ------------------------------------------------
# 8. MOST ACTIVE INVESTORS
# ------------------------------------------------
top_investors = df['investors'].value_counts().head(10)

plt.figure()
top_investors.plot(kind='bar')
plt.title("Most Active Investors")
plt.show()

# ------------------------------------------------
# 9. INVESTMENT TYPE ANALYSIS
# ------------------------------------------------
investment_type = df.groupby('investment_type')['amount'].sum().sort_values(ascending=False)

plt.figure()
investment_type.plot(kind='bar')
plt.title("Funding by Investment Type")
plt.show()

# ------------------------------------------------
# 10. TOP STARTUPS BY FUNDING
# ------------------------------------------------
top_startups = df.groupby('startup')['amount'].sum().sort_values(ascending=False).head(10)

plt.figure()
top_startups.plot(kind='bar')
plt.title("Top Funded Startups")
plt.show()

# ------------------------------------------------
# 11. MACHINE LEARNING MODEL (Predict Funding)
# ------------------------------------------------
ml_df = df[['industry','investment_type','amount']].copy()

le = LabelEncoder()
for col in ['industry','investment_type']:
    ml_df[col] = le.fit_transform(ml_df[col].astype(str))

X = ml_df.drop('amount', axis=1)
y = ml_df['amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("\nModel R2 Score:", r2_score(y_test, pred))

# Feature importance
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind='barh', title="Feature Importance")
plt.show()

# ------------------------------------------------
# 12. EXPORT CLEAN DATA FOR POWER BI
# ------------------------------------------------
output_path = r"D:\indian-startup-funding-analysis\archive(4)\clean_startup_funding.csv"
df.to_csv(output_path, index=False)

print("\nPROJECT COMPLETED SUCCESSFULLY 🚀")
print("Clean dataset saved for Power BI")
