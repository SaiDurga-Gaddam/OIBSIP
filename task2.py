# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
data = pd.read_csv("Unemployment in India.csv")

# Displaying first few rows of the data
print("First 5 rows of the dataset:")
print(data.head())

# Checking basic info about the dataset
print("\nDataset Info:")
print(data.info())

# Checking for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Cleaning column names by stripping whitespace
data.columns = data.columns.str.strip()

# Converting Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Checking for duplicates and removing them
data = data.drop_duplicates()

# Unemployment rate by region (average)
plt.figure(figsize=(12,6))
sns.barplot(x='Region', y='Estimated Unemployment Rate (%)', data=data)
plt.xticks(rotation=90)
plt.title('Average Unemployment Rate by Region')
plt.tight_layout()
plt.show()

# Unemployment trend over time (overall average)
trend = data.groupby('Date')['Estimated Unemployment Rate (%)'].mean().reset_index()

plt.figure(figsize=(12,6))
plt.plot(trend['Date'], trend['Estimated Unemployment Rate (%)'], color='blue', marker='o')
plt.title('Unemployment Rate Over Time (India Average)')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Unemployment rate by state/region (latest date)
latest_date = data['Date'].max()
latest_data = data[data['Date'] == latest_date]

plt.figure(figsize=(14,6))
sns.barplot(x='Region', y='Estimated Unemployment Rate (%)', data=latest_data)
plt.xticks(rotation=90)
plt.title(f'Unemployment Rate by Region on {latest_date.date()}')
plt.tight_layout()
plt.show()

# Heatmap to see correlation between numerical columns
plt.figure(figsize=(8,6))
sns.heatmap(data[['Estimated Unemployment Rate (%)', 'Estimated Employed',
                  'Estimated Labour Participation Rate (%)']].corr(), annot=True, cmap='YlGnBu')
plt.title('Correlation between Factors')
plt.tight_layout()
plt.show()

# Done!
print("Analysis completed.")
