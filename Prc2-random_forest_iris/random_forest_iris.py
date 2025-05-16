from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Iris Dataset
data = load_iris()
X = data.data
y = data.target

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and Train the Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make Predictions
predictions = model.predict(X_test)

# Evaluate the Model
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
