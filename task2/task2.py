import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Sample data
data = {'age': [30, 35, 40, 45, 50, 55, 60, 65, 70],
        'balance': [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000],
        'subscribed': [0, 0, 0, 0, 1, 1, 1, 1, 1]}  # 1: subscribed, 0: not subscribed

# Creating a DataFrame
df = pd.DataFrame(data)

# Visualizing the data
plt.scatter(df['age'], df['balance'], c=df['subscribed'], cmap='bwr', edgecolors='k')
plt.title('Subscription Status')
plt.xlabel('Age')
plt.ylabel('Balance')
plt.grid(True)
plt.show()

# Splitting the data into train and test sets
X = df[['age', 'balance']]
y = df['subscribed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
