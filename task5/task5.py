import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the data
data = pd.read_csv('spam-data.csv')

# Separate features and target variable
X = data.drop(columns=['Class'])
y = data['Class']

# Step 2: Build and train the logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Step 3: Parse the emails.txt file and extract email features
def extract_features(email):
    # Split the email into lines
    lines = email.split('\n')
    subject = lines[0].split(': ')[1]
    num_links = len(lines[2].split('|'))
    num_words = len(lines[3].split())
    num_capitalized = sum(1 for word in lines[3].split() if word.isupper())
    return [num_words, num_links, num_capitalized, subject.lower().count('free')]

# Read emails from emails.txt
with open('emails.txt', 'r') as file:
    emails = file.read().split('----------------\n')[:-1]

# Step 4: Check emails for spam and print results
print("\nResults for email spam detection:")
for i, email in enumerate(emails, 1):
    features = extract_features(email)
    prediction = model.predict([features])[0]
    print(f"Email {i}: {'Spam' if prediction == 1 else 'Not Spam'}")

# Step 5: Analysis of the spam-data.csv file
# Feature Importance Analysis
feature_importance = pd.Series(model.coef_[0], index=X.columns)
print("\nFeature Importance Analysis:")
print(feature_importance)

# Features with coefficient close to zero are less important for spam detection
unimportant_features = feature_importance[abs(feature_importance) < 0.1].index.tolist()
print("\nFeatures that are not important for spam detection:")
print(unimportant_features)

#Output:
#Training Accuracy: 0.9292035398230089
#Testing Accuracy: 0.9310344827586207

#Results for email spam detection:
#Email 1: Not Spam
#Email 2: Not Spam

#Feature Importance Analysis:
#Number of Words               -0.060083
#Number of Links                0.902070
#Number of Capitalized Words   -0.459948
#Number of Spam Words           1.160161
#dtype: float64