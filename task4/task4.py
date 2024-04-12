# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load the data
data = pd.read_csv('spam-data.csv')

# Split the data into features (X) and target variable (y)
X = data.drop('Class', axis=1)  # Features
y = data['Class']               # Target variable

# Divide the data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
#Output Confusion Matrix:
#[[16  1]
# [ 1 11]]