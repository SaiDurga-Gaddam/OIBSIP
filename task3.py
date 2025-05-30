import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('car data.csv')
print("Initial Data Preview:")
print(df.head())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Graph 1: Selling Price Distribution
sns.histplot(df['Selling_Price'], kde=True)
plt.title("Selling Price Distribution")
plt.xlabel("Selling Price")
plt.ylabel("Count")
plt.show()

#  Graph 2: Boxplot of Price vs Year 
sns.boxplot(x='Year', y='Selling_Price', data=df)
plt.title("Car Price by Year")
plt.xticks(rotation=45)
plt.show()

#  Graph 3: Correlation Heatmap 
# Convert categorical to dummy before heatmap
df_encoded = pd.get_dummies(df, drop_first=True)
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Data Preprocessing 
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Model Training 
model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

#  Graph 4: Actual vs Predicted 
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
sns.scatterplot(x='Actual', y='Predicted', data=results)
plt.plot([0, max(results['Actual'])], [0, max(results['Actual'])], 'r--')
plt.title("Actual vs Predicted Selling Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

# Save Model and Scaler
joblib.dump(model, 'car_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and Scaler saved as 'car_price_model.pkl' and 'scaler.pkl'")
