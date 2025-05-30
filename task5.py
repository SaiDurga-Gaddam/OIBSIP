import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv('Advertising.csv')

# Step 2: Explore the data
print(data.head())
print(data.describe())
print(data.info())

# Step 3: Visualize relationships
sns.pairplot(data)
plt.show()

# Step 4: Prepare features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Step 5: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the model on test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R^2 Score: {r2:.3f}")

# Step 8: Predict sales dynamically based on user input
def predict_sales(tv, radio, newspaper):
    features = [[tv, radio, newspaper]]
    prediction = model.predict(features)
    return prediction[0]

try:
    tv_input = float(input("Enter TV advertising spend: "))
    radio_input = float(input("Enter Radio advertising spend: "))
    newspaper_input = float(input("Enter Newspaper advertising spend: "))
    
    sales_pred = predict_sales(tv_input, radio_input, newspaper_input)
    print(f"Predicted Sales: {sales_pred:.2f}")
except ValueError:
    print("Invalid input. Please enter numeric values.")
