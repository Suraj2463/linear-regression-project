# ==============================
# 🚗 CAR PRICE PREDICTION PROJECT
# ==============================

# 📦 Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

# ==============================
# 📥 STEP 1: Load the Dataset
# ==============================
df = pd.read_csv('cardekho (6).csv')

print("Initial Data Snapshot:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# ==============================
# 🧹 STEP 2: Data Cleaning
# ==============================

# Extract brand name from 'name' column
df['brand'] = df['name'].str.split().str.get(0)

# Convert 'max_power' to float
df['max_power'] = df['max_power'].str.split().str.get(0).astype('float')

# Ensure 'year' is integer
df['year'] = df['year'].astype('int')

# Drop unnecessary column
df.drop(columns=['name'], inplace=True)

print("\nAfter Cleaning:")
print(df.info())

# ==============================
# 🧩 STEP 3: Handling Missing Values
# ==============================

print("\nMissing Values Before:")
print(df.isnull().sum())

df['mileage(km/ltr/kg)'] = df['mileage(km/ltr/kg)'].fillna(df['mileage(km/ltr/kg)'].mean())
df['engine'] = df['engine'].fillna(df['engine'].mean())
df['seats'] = df['seats'].fillna(df['seats'].mode()[0])
df['max_power'] = df['max_power'].fillna(df['max_power'].mode()[0])

print("\nMissing Values After:")
print(df.isnull().sum())

# ==============================
# 🔁 STEP 4: Remove Duplicates
# ==============================
print("\nDuplicate Rows Before:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicate Rows After:", df.duplicated().sum())

# ==============================
# 🔠 STEP 5: Encoding Categorical Variables
# ==============================
le = LabelEncoder()
cat_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']

for col in cat_columns:
    df[col] = le.fit_transform(df[col])

print("\nAfter Label Encoding:")
print(df.info())

# ==============================
# 📊 STEP 6: Reducing Noise (Grouping Rare Brands)
# ==============================
brands_to_remove = df['brand'].value_counts()[df['brand'].value_counts() < 30].index.tolist()
df['brand'] = df['brand'].apply(lambda x: 0 if x in brands_to_remove else x)  # 0 → represents rare brands

print("\nTop Brand Frequencies:")
print(df['brand'].value_counts().head())

# ==============================
# ⚖️ STEP 7: Feature Scaling
# ==============================
scaler = MinMaxScaler()
cols_to_scale = ['year', 'km_driven', 'fuel', 'seller_type', 'transmission',
                 'owner', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats', 'brand']

df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print("\nData After Normalization:")
print(df.head())

# ==============================
# 🧠 STEP 8: Splitting the Dataset
# ==============================
X = df.drop(columns=['selling_price'])
y = df['selling_price']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, shuffle=True)

# ==============================
# 🤖 STEP 9: Model Training & Evaluation
# ==============================
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=3),
    'Random Forest': RandomForestRegressor(n_estimators=200),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor()
}

results = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    results[name] = {
        'R² Score': r2_score(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred)
    }

print("\n📈 Model Performance Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: {metrics}")

# ==============================
# 🔍 STEP 10: Individual Prediction Example
# ==============================
sample = df.loc[4].drop('selling_price')
xg_pred = models['XGBoost'].predict(sample.values.reshape(1, -1))
rf_pred = models['Random Forest'].predict(sample.values.reshape(1, -1))

print("\nPredictions for Sample Row 4:")
print("Actual Price:", df.loc[4, 'selling_price'])
print("XGBoost Predicted:", xg_pred)
print("Random Forest Predicted:", rf_pred)

# ==============================
# 📊 STEP 11: Visualization
# ==============================

# R² Comparison
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), [r['R² Score'] for r in results.values()], color='orange')
plt.title("Model Comparison by R² Score")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.show()

# Selling Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['selling_price'], kde=True, color='skyblue')
plt.title("Distribution of Selling Prices")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Actual vs Predicted (for XGBoost)
y_pred = models['XGBoost'].predict(x_test)
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='black', label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Prices (XGBoost)")
plt.legend()
plt.show()


