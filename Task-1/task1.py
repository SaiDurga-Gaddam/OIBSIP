import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv('Iris.csv')

# Drop unnecessary column
df.drop('Id', axis=1, inplace=True)

# Basic dataset overview
print("First few records:")
print(df.head())
print("\nDataset summary:")
print(df.info())
print("\nClass distribution:")
print(df['Species'].value_counts())

# Data visualization
sns.pairplot(df, hue='Species')
plt.suptitle("Iris Feature Pairplot", y=1.02)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Prepare data
X = df.drop('Species', axis=1)
y = df['Species']

# Encode species labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
Ry_pred = clf.predict(X_test)

# Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save trained model
joblib.dump(clf, 'iris_classifier.pkl')
print("\nTrained model saved as 'iris_classifier.pkl'")
